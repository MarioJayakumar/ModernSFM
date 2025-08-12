"""
GPU-accelerated bundle adjustment using PyTorch optimization.
Implements global and local bundle adjustment for robust 3D reconstruction refinement.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import cv2

logger = logging.getLogger(__name__)

@dataclass
class BundleAdjustmentResult:
    """Results from bundle adjustment optimization."""
    optimized_points_3d: np.ndarray
    optimized_poses: List[Dict[str, np.ndarray]]
    optimized_intrinsics: Optional[np.ndarray]
    initial_rmse: float
    final_rmse: float
    num_iterations: int
    convergence_reason: str
    inlier_ratio: float
    optimization_time: float

class ReprojectionLoss(nn.Module):
    """
    Robust reprojection loss with Huber loss for outlier handling.
    """
    
    def __init__(self, huber_delta: float = 1.0, use_huber: bool = True):
        super().__init__()
        self.huber_delta = huber_delta
        self.use_huber = use_huber
        self.huber_loss = nn.HuberLoss(delta=huber_delta, reduction='none')
        
    def forward(self, predicted_points: torch.Tensor, observed_points: torch.Tensor, 
                weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute reprojection loss.
        
        Args:
            predicted_points: [N, 2] predicted 2D points
            observed_points: [N, 2] observed 2D points  
            weights: [N] optional weights for each observation
            
        Returns:
            Scalar loss value
        """
        # Compute residuals
        residuals = predicted_points - observed_points  # [N, 2]
        
        if self.use_huber:
            # Apply Huber loss to each coordinate
            losses = self.huber_loss(predicted_points, observed_points)  # [N, 2]
            losses = losses.sum(dim=1)  # [N] - sum x,y components
        else:
            # Standard L2 loss
            losses = (residuals ** 2).sum(dim=1)  # [N]
            
        # Apply weights if provided
        if weights is not None:
            losses = losses * weights
            
        return losses.mean()

class CameraParameterization(nn.Module):
    """
    Camera pose parameterization using axis-angle rotation and translation.
    """
    
    def __init__(self, initial_poses: List[Dict[str, torch.Tensor]], device: torch.device):
        super().__init__()
        self.device = device
        self.num_cameras = len(initial_poses)
        
        # Convert poses to axis-angle + translation parameterization
        rotations = []
        translations = []
        
        for pose in initial_poses:
            R = pose['R'].to(device)  # [3, 3]
            t = pose['t'].to(device)  # [3, 1] or [3]
            
            # Convert rotation matrix to axis-angle
            rvec, _ = cv2.Rodrigues(R.cpu().numpy())
            rvec = torch.from_numpy(rvec.flatten()).float().to(device)
            
            # Ensure translation is [3]
            if t.dim() == 2:
                t = t.flatten()
                
            rotations.append(rvec)
            translations.append(t)
            
        # Create learnable parameters
        self.rotations = nn.Parameter(torch.stack(rotations))  # [N, 3]
        self.translations = nn.Parameter(torch.stack(translations))  # [N, 3]
        
    def forward(self) -> List[Dict[str, torch.Tensor]]:
        """
        Convert parameterized poses back to rotation matrices and translations.
        
        Returns:
            List of pose dictionaries with 'R' and 't' keys
        """
        poses = []
        
        for i in range(self.num_cameras):
            # Convert axis-angle to rotation matrix
            rvec = self.rotations[i].detach().cpu().numpy()
            R, _ = cv2.Rodrigues(rvec)
            R = torch.from_numpy(R).float().to(self.device)
            
            t = self.translations[i]
            
            poses.append({
                'R': R,
                't': t.unsqueeze(1) if t.dim() == 1 else t  # Ensure [3, 1]
            })
            
        return poses

class BundleAdjuster:
    """
    GPU-accelerated bundle adjustment using PyTorch optimization.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Optimization parameters
        self.max_iterations = config.bundle_adjustment.max_iterations
        self.function_tolerance = config.bundle_adjustment.function_tolerance
        self.gradient_tolerance = config.bundle_adjustment.gradient_tolerance
        self.parameter_tolerance = config.bundle_adjustment.parameter_tolerance
        
        # Robust loss parameters
        self.huber_delta = getattr(config.bundle_adjustment, 'huber_delta', 1.0)
        self.use_huber_loss = getattr(config.bundle_adjustment, 'use_huber_loss', True)
        
        # Optimization settings
        self.learning_rate = getattr(config.bundle_adjustment, 'learning_rate', 1e-4)
        self.optimizer_type = getattr(config.bundle_adjustment, 'optimizer', 'adam')
        
        logger.info(f"Initialized BundleAdjuster on device: {self.device}")
        
    def optimize_reconstruction(self, points_3d: np.ndarray, poses: List[Dict[str, np.ndarray]], 
                              intrinsics: np.ndarray, observations: List[Dict]) -> BundleAdjustmentResult:
        """
        Global bundle adjustment optimizing all cameras and 3D points.
        
        Args:
            points_3d: [N, 3] 3D points
            poses: List of camera poses with 'R' and 't' keys
            intrinsics: [3, 3] camera intrinsic matrix
            observations: List of observation dictionaries with point correspondences
            
        Returns:
            BundleAdjustmentResult with optimized parameters and statistics
        """
        import time
        start_time = time.time()
        
        logger.info(f"Starting global bundle adjustment with {len(points_3d)} points and {len(poses)} cameras")
        
        # Convert to PyTorch tensors
        points_3d_torch = torch.from_numpy(points_3d).float().to(self.device)
        intrinsics_torch = torch.from_numpy(intrinsics).float().to(self.device)
        
        # Convert poses to PyTorch format
        poses_torch = []
        for pose in poses:
            poses_torch.append({
                'R': torch.from_numpy(pose['R']).float(),
                't': torch.from_numpy(pose['t']).float()
            })
        
        # Create optimizable parameters
        points_param = nn.Parameter(points_3d_torch.clone())
        camera_param = CameraParameterization(poses_torch, self.device)
        
        # Optionally optimize intrinsics
        optimize_intrinsics = getattr(self.config.bundle_adjustment, 'optimize_intrinsics', False)
        if optimize_intrinsics:
            # Parameterize intrinsics as [fx, fy, cx, cy]
            intrinsics_param = nn.Parameter(torch.tensor([
                intrinsics[0, 0], intrinsics[1, 1], 
                intrinsics[0, 2], intrinsics[1, 2]
            ]).float().to(self.device))
        else:
            intrinsics_param = None
        
        # Setup optimizer
        params = [points_param] + list(camera_param.parameters())
        if intrinsics_param is not None:
            params.append(intrinsics_param)
            
        if self.optimizer_type.lower() == 'adam':
            optimizer = optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == 'lbfgs':
            optimizer = optim.LBFGS(params, lr=self.learning_rate, max_iter=20)
        else:
            optimizer = optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        
        # Setup loss function
        loss_fn = ReprojectionLoss(self.huber_delta, self.use_huber_loss)
        
        # Compute initial RMSE
        initial_rmse = self._compute_rmse(points_param, camera_param(), intrinsics_torch, observations)
        logger.info(f"Initial RMSE: {initial_rmse:.4f} pixels")
        
        # Optimization loop
        prev_loss = float('inf')
        convergence_reason = "max_iterations"
        
        for iteration in range(self.max_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Get current camera poses
                current_poses = camera_param()
                
                # Get current intrinsics
                if intrinsics_param is not None:
                    current_K = torch.zeros(3, 3).to(self.device)
                    current_K[0, 0] = intrinsics_param[0]  # fx
                    current_K[1, 1] = intrinsics_param[1]  # fy
                    current_K[0, 2] = intrinsics_param[2]  # cx
                    current_K[1, 2] = intrinsics_param[3]  # cy
                    current_K[2, 2] = 1.0
                else:
                    current_K = intrinsics_torch
                
                # Compute reprojection loss
                total_loss = self._compute_reprojection_loss(
                    points_param, current_poses, current_K, observations, loss_fn
                )
                
                total_loss.backward()
                return total_loss
            
            # Optimization step
            if self.optimizer_type.lower() == 'lbfgs':
                loss = optimizer.step(closure)
                if loss is None:
                    loss = closure()
            else:
                loss = closure()
                optimizer.step()
            
            current_loss = float(loss.item())
            
            # Check convergence
            if iteration > 0:
                loss_change = abs(prev_loss - current_loss)
                if loss_change < self.function_tolerance:
                    convergence_reason = "function_tolerance"
                    break
                    
                # Check gradient norm
                total_grad_norm = 0.0
                for param in params:
                    if param.grad is not None:
                        total_grad_norm += param.grad.norm().item() ** 2
                total_grad_norm = total_grad_norm ** 0.5
                
                if total_grad_norm < self.gradient_tolerance:
                    convergence_reason = "gradient_tolerance"
                    break
            
            prev_loss = current_loss
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Loss = {current_loss:.6f}")
        
        # Compute final RMSE
        final_poses = camera_param()
        final_K = intrinsics_torch
        if intrinsics_param is not None:
            final_K = torch.zeros(3, 3).to(self.device)
            final_K[0, 0] = intrinsics_param[0]
            final_K[1, 1] = intrinsics_param[1]
            final_K[0, 2] = intrinsics_param[2]
            final_K[1, 2] = intrinsics_param[3]
            final_K[2, 2] = 1.0
            
        final_rmse = self._compute_rmse(points_param, final_poses, final_K, observations)
        
        # Compute inlier ratio (points with reprojection error < 2 pixels)
        inlier_ratio = self._compute_inlier_ratio(points_param, final_poses, final_K, observations, threshold=2.0)
        
        optimization_time = time.time() - start_time
        
        logger.info(f"Bundle adjustment completed in {optimization_time:.2f}s")
        logger.info(f"Final RMSE: {final_rmse:.4f} pixels (improvement: {initial_rmse - final_rmse:.4f})")
        logger.info(f"Inlier ratio: {inlier_ratio:.3f}")
        logger.info(f"Convergence reason: {convergence_reason}")
        
        # Convert results back to numpy
        optimized_points_3d = points_param.detach().cpu().numpy()
        optimized_poses = []
        for pose in final_poses:
            optimized_poses.append({
                'R': pose['R'].detach().cpu().numpy(),
                't': pose['t'].detach().cpu().numpy()
            })
        
        optimized_intrinsics = None
        if intrinsics_param is not None:
            optimized_intrinsics = final_K.detach().cpu().numpy()
        
        return BundleAdjustmentResult(
            optimized_points_3d=optimized_points_3d,
            optimized_poses=optimized_poses,
            optimized_intrinsics=optimized_intrinsics,
            initial_rmse=initial_rmse,
            final_rmse=final_rmse,
            num_iterations=iteration + 1,
            convergence_reason=convergence_reason,
            inlier_ratio=inlier_ratio,
            optimization_time=optimization_time
        )
    
    def local_bundle_adjustment(self, points_3d: np.ndarray, poses: List[Dict[str, np.ndarray]], 
                               intrinsics: np.ndarray, observations: List[Dict],
                               window_size: int = 10) -> BundleAdjustmentResult:
        """
        Local bundle adjustment with sliding window for efficiency.
        
        Args:
            points_3d: [N, 3] 3D points
            poses: List of camera poses
            intrinsics: [3, 3] camera intrinsic matrix
            observations: List of observation dictionaries
            window_size: Number of cameras to optimize simultaneously
            
        Returns:
            BundleAdjustmentResult with optimized parameters
        """
        logger.info(f"Starting local bundle adjustment with window size {window_size}")
        
        if len(poses) <= window_size:
            # If we have fewer cameras than window size, do global BA
            return self.optimize_reconstruction(points_3d, poses, intrinsics, observations)
        
        # For now, implement a simplified version that optimizes the last window_size cameras
        # In a full implementation, this would slide the window across all cameras
        
        # Select last window_size cameras and their associated points
        start_idx = max(0, len(poses) - window_size)
        local_poses = poses[start_idx:]
        
        # Filter observations to only include those visible in the local window
        local_observations = []
        local_point_indices = set()
        
        for obs in observations:
            if obs['camera_id'] >= start_idx:
                local_observations.append({
                    'camera_id': obs['camera_id'] - start_idx,  # Adjust camera index
                    'point_id': obs['point_id'],
                    'point_2d': obs['point_2d']
                })
                local_point_indices.add(obs['point_id'])
        
        # Extract local 3D points
        local_point_indices = sorted(list(local_point_indices))
        local_points_3d = points_3d[local_point_indices]
        
        # Adjust point IDs in observations
        point_id_mapping = {old_id: new_id for new_id, old_id in enumerate(local_point_indices)}
        for obs in local_observations:
            obs['point_id'] = point_id_mapping[obs['point_id']]
        
        # Run bundle adjustment on local subset
        result = self.optimize_reconstruction(local_points_3d, local_poses, intrinsics, local_observations)
        
        # Update the original arrays with optimized values
        optimized_points_3d = points_3d.copy()
        optimized_points_3d[local_point_indices] = result.optimized_points_3d
        
        optimized_poses = poses.copy()
        optimized_poses[start_idx:] = result.optimized_poses
        
        # Return result with full arrays
        result.optimized_points_3d = optimized_points_3d
        result.optimized_poses = optimized_poses
        
        return result
    
    def _compute_reprojection_loss(self, points_3d: torch.Tensor, poses: List[Dict[str, torch.Tensor]], 
                                  intrinsics: torch.Tensor, observations: List[Dict],
                                  loss_fn: ReprojectionLoss) -> torch.Tensor:
        """
        Compute reprojection loss for all observations.
        """
        all_predicted = []
        all_observed = []
        
        for obs in observations:
            camera_id = obs['camera_id']
            point_id = obs['point_id']
            observed_2d = torch.from_numpy(obs['point_2d']).float().to(self.device)
            
            # Skip invalid indices
            if point_id >= len(points_3d) or camera_id >= len(poses):
                continue
                
            # Ensure observed_2d is exactly [2] shape
            if observed_2d.dim() > 1:
                observed_2d = observed_2d.squeeze()
            if observed_2d.numel() != 2:
                continue  # Skip malformed observations
                
            # Get 3D point and camera pose
            point_3d = points_3d[point_id]  # [3]
            pose = poses[camera_id]
            R = pose['R']  # [3, 3]
            t = pose['t']  # [3, 1] or [3]
            
            # Ensure t is [3, 1]
            if t.dim() == 1:
                t = t.unsqueeze(1)
            
            # Transform world point to camera coordinates
            point_3d_cam = R @ point_3d.unsqueeze(1) + t  # [3, 1]
            point_3d_cam = point_3d_cam.squeeze(1)  # [3]
            
            # Only project points in front of camera with reasonable depth
            if point_3d_cam[2] > 0.1:  # Minimum depth threshold
                # Perspective projection: P = K * X_cam
                point_2d_hom = intrinsics @ point_3d_cam.unsqueeze(1)  # [3, 1]
                point_2d = point_2d_hom[:2] / (point_2d_hom[2] + 1e-8)  # [2] with numerical stability
                point_2d = point_2d.squeeze()  # Ensure [2] shape
                
                all_predicted.append(point_2d)
                all_observed.append(observed_2d)
        
        if len(all_predicted) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # Stack all predictions and observations
        predicted_batch = torch.stack(all_predicted)  # [N, 2]
        observed_batch = torch.stack(all_observed)    # [N, 2]
        
        # Compute loss
        return loss_fn(predicted_batch, observed_batch)
    
    def _compute_rmse(self, points_3d: torch.Tensor, poses: List[Dict[str, torch.Tensor]], 
                     intrinsics: torch.Tensor, observations: List[Dict]) -> float:
        """
        Compute root mean square reprojection error.
        """
        total_error = 0.0
        num_observations = 0
        
        with torch.no_grad():
            # Detach tensors to avoid gradient computation
            points_3d_detached = points_3d.detach()
            intrinsics_detached = intrinsics.detach()
            
            for obs in observations:
                camera_id = obs['camera_id']
                point_id = obs['point_id']
                
                # Skip invalid indices
                if point_id >= len(points_3d_detached) or camera_id >= len(poses):
                    continue
                    
                observed_2d = torch.from_numpy(obs['point_2d']).float().to(self.device)
                
                # Get 3D point and camera pose (detached)
                point_3d = points_3d_detached[point_id]
                pose = poses[camera_id]
                R = pose['R'].detach()
                t = pose['t'].detach()
                
                if t.dim() == 1:
                    t = t.unsqueeze(1)
                
                # Transform to camera coordinates
                point_3d_cam = R @ point_3d.unsqueeze(1) + t
                point_3d_cam = point_3d_cam.squeeze(1)
                
                # Only compute error for points in front of camera
                if point_3d_cam[2] > 0.1:
                    point_2d_hom = intrinsics_detached @ point_3d_cam.unsqueeze(1)
                    point_2d = point_2d_hom[:2] / (point_2d_hom[2] + 1e-8)
                    
                    # Compute squared euclidean error
                    error = ((point_2d - observed_2d) ** 2).sum()
                    total_error += error.item()
                    num_observations += 1
        
        if num_observations == 0:
            return 0.0
        return (total_error / num_observations) ** 0.5
    
    def _compute_inlier_ratio(self, points_3d: torch.Tensor, poses: List[Dict[str, torch.Tensor]], 
                             intrinsics: torch.Tensor, observations: List[Dict], 
                             threshold: float = 2.0) -> float:
        """
        Compute ratio of observations with reprojection error below threshold.
        """
        num_inliers = 0
        num_observations = 0
        
        with torch.no_grad():
            # Detach tensors to avoid gradient computation
            points_3d_detached = points_3d.detach()
            intrinsics_detached = intrinsics.detach()
            
            for obs in observations:
                camera_id = obs['camera_id']
                point_id = obs['point_id']
                
                # Skip invalid indices
                if point_id >= len(points_3d_detached) or camera_id >= len(poses):
                    continue
                    
                observed_2d = torch.from_numpy(obs['point_2d']).float().to(self.device)
                
                # Get 3D point and camera pose (detached)
                point_3d = points_3d_detached[point_id]
                pose = poses[camera_id]
                R = pose['R'].detach()
                t = pose['t'].detach()
                
                if t.dim() == 1:
                    t = t.unsqueeze(1)
                
                # Transform to camera coordinates
                point_3d_cam = R @ point_3d.unsqueeze(1) + t
                point_3d_cam = point_3d_cam.squeeze(1)
                
                # Only compute error for points in front of camera
                if point_3d_cam[2] > 0.1:
                    point_2d_hom = intrinsics_detached @ point_3d_cam.unsqueeze(1)
                    point_2d = point_2d_hom[:2] / (point_2d_hom[2] + 1e-8)
                    
                    # Compute reprojection error (Euclidean distance)
                    error = ((point_2d - observed_2d) ** 2).sum() ** 0.5
                    if error < threshold:
                        num_inliers += 1
                    num_observations += 1
        
        return num_inliers / max(num_observations, 1)
