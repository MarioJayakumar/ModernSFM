"""
Classical Pose Estimation Module for Modern SfM Pipeline

This module implements robust camera pose estimation using classical geometric
methods with COLMAP and OpenCV integration for maximum reliability.

Author: ModernSFM Pipeline
Date: July 25th, 2025
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path

try:
    import pycolmap
    COLMAP_AVAILABLE = True
except ImportError:
    COLMAP_AVAILABLE = False
    logging.warning("PyColmap not available. Using OpenCV-only pose estimation.")


class PoseEstimator:
    """
    Robust pose estimation using classical geometric methods.
    
    Supports both COLMAP and OpenCV solvers with automatic fallback
    for maximum reliability across different scenarios.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.pose_config = config.get('pose_estimation', {})
        
        # RANSAC parameters
        self.ransac_config = self.pose_config.get('ransac', {})
        self.ransac_threshold = self.ransac_config.get('threshold', 1.0)
        self.ransac_confidence = self.ransac_config.get('confidence', 0.9999)
        self.ransac_max_iters = self.ransac_config.get('max_iterations', 10000)
        
        # Method preference
        self.preferred_method = self.pose_config.get('method', 'colmap')
        
        logging.info(f"Initialized PoseEstimator with method: {self.preferred_method}")
        logging.info(f"COLMAP available: {COLMAP_AVAILABLE}")
    
    def estimate_two_view_geometry(self,
                                 points1: np.ndarray,
                                 points2: np.ndarray,
                                 K1: np.ndarray,
                                 K2: np.ndarray,
                                 method: Optional[str] = None) -> Dict:
        """
        Estimate relative pose between two views.
        
        Args:
            points1, points2: [N, 2] corresponding points in pixel coordinates
            K1, K2: [3, 3] camera intrinsic matrices
            method: 'colmap', 'opencv', or None (use config default)
            
        Returns:
            Dictionary containing:
                - R: [3, 3] rotation matrix from camera 1 to camera 2
                - t: [3, 1] translation vector from camera 1 to camera 2
                - essential_matrix: [3, 3] essential matrix
                - fundamental_matrix: [3, 3] fundamental matrix
                - inlier_mask: [N] boolean mask of inliers
                - num_inliers: int number of inliers
                - inlier_ratio: float ratio of inliers
                - method_used: str method that was successful
                - success: bool whether estimation succeeded
        """
        def _empty_result():
            return {
                'R': None,
                't': None,
                'essential_matrix': None,
                'fundamental_matrix': None,
                'inlier_mask': None,
                'num_inliers': 0,
                'inlier_ratio': 0.0,
                'method_used': 'none',
                'success': False
            }

        candidate_results = []
        min_inliers = self.pose_config.get('min_inliers', 0)
        
        if len(points1) < 8:
            logging.warning(f"Insufficient correspondences: {len(points1)} < 8")
            return result
        
        # Determine method to use
        use_method = method or self.preferred_method
        min_rotation = float(self.pose_config.get('min_rotation_deg', 0.0))
        min_parallax = float(self.pose_config.get('min_parallax_deg', 0.0))
        parallax_deg = self._estimate_parallax(points1, points2)
        if parallax_deg is not None:
            logging.debug("Estimated parallax for pair: %.2f°", parallax_deg)
        
        # Try COLMAP first if available and requested
        if use_method == 'colmap' and COLMAP_AVAILABLE:
            try:
                colmap_result = self._estimate_pose_colmap(points1, points2, K1, K2)
                if colmap_result['success'] and colmap_result['num_inliers'] >= min_inliers:
                    colmap_result['method_used'] = 'colmap'
                    candidate_results.append(colmap_result)
                    logging.info(f"COLMAP pose estimation successful: {colmap_result['num_inliers']}/{len(points1)} inliers")
                else:
                    logging.info(
                        "COLMAP pose estimation discarded: %s",
                        "insufficient inliers" if colmap_result.get('success') else "failure"
                    )
            except Exception as e:
                logging.warning(f"COLMAP pose estimation error: {e}, falling back to OpenCV")
        
        # Try OpenCV method
        try:
            opencv_result = self._estimate_pose_opencv(points1, points2, K1, K2)
            if opencv_result['success'] and opencv_result['num_inliers'] >= min_inliers:
                opencv_result['method_used'] = 'opencv'
                candidate_results.append(opencv_result)
                logging.info(f"OpenCV pose estimation successful: {opencv_result['num_inliers']}/{len(points1)} inliers")
            else:
                logging.info(
                    "OpenCV pose estimation discarded: %s",
                    "insufficient inliers" if opencv_result.get('success') else "failure"
                )
        except Exception as e:
            logging.error(f"OpenCV pose estimation failed: {e}")

        filtered_candidates = candidate_results
        if candidate_results and (min_rotation > 0.0 or min_parallax > 0.0):
            filtered_candidates = []
            for res in candidate_results:
                rot_ok = res.get('rotation_angle_deg', 0.0) >= min_rotation
                parallax_ok = parallax_deg is None or parallax_deg >= min_parallax
                if rot_ok and parallax_ok:
                    filtered_candidates.append(res)

            if not filtered_candidates and candidate_results:
                logging.debug(
                    "All pose candidates rejected by rotation/parallax thresholds (min_rot=%.2f°, min_parallax=%.2f°); keeping best available.",
                    min_rotation,
                    min_parallax
                )
                filtered_candidates = candidate_results

        if filtered_candidates:
            def _score(result: Dict) -> Tuple[float, int, int]:
                return (
                    result.get('rotation_angle_deg', 0.0),
                    result.get('positive_depth_count', 0),
                    result.get('num_inliers', 0)
                )
            
            best = max(filtered_candidates, key=_score)
            return best
        
        logging.error("All pose estimation methods failed")
        return _empty_result()
    
    def _estimate_pose_colmap(self,
                            points1: np.ndarray,
                            points2: np.ndarray,
                            K1: np.ndarray,
                            K2: np.ndarray) -> Dict:
        """Estimate pose using COLMAP's robust solvers."""
        try:
            # Convert points to correct format for COLMAP
            points1 = points1.astype(np.float64)
            points2 = points2.astype(np.float64)
            
            # Create COLMAP camera objects
            # Assume SIMPLE_RADIAL camera model with focal length from K matrix
            fx1, fy1 = K1[0, 0], K1[1, 1]
            cx1, cy1 = K1[0, 2], K1[1, 2]
            fx2, fy2 = K2[0, 0], K2[1, 1]
            cx2, cy2 = K2[0, 2], K2[1, 2]
            
            # Create camera objects using the new COLMAP API
            camera1 = pycolmap.Camera(
                model='SIMPLE_PINHOLE',
                width=int(2 * cx1),  # Estimate image width
                height=int(2 * cy1), # Estimate image height  
                params=[fx1, cx1, cy1]
            )
            camera2 = pycolmap.Camera(
                model='SIMPLE_PINHOLE',
                width=int(2 * cx2),
                height=int(2 * cy2),
                params=[fx2, cx2, cy2]
            )
            
            # COLMAP RANSAC options
            ransac_options = pycolmap.RANSACOptions()
            ransac_options.max_error = self.ransac_threshold
            ransac_options.confidence = self.ransac_confidence
            ransac_options.max_num_trials = self.ransac_max_iters
            ransac_options.min_inlier_ratio = 0.1
            
            # Estimate essential matrix using new API
            result = pycolmap.estimate_essential_matrix(
                points1, points2, camera1, camera2, ransac_options
            )
            
            if result is None or 'E' not in result:
                return {'success': False}
            
            E = result['E']
            inlier_mask = result.get('inliers', np.ones(len(points1), dtype=bool))
            
            # Normalize points for pose recovery
            points1_norm = self._normalize_points(points1, K1)
            points2_norm = self._normalize_points(points2, K2)
            
        except Exception as e:
            logging.warning(f"COLMAP pose estimation error: {e}, falling back to OpenCV")
            return {'success': False}
        
        # Recover pose from essential matrix
        R, t, pose_inliers = self._recover_pose_from_essential(
            E, points1_norm[inlier_mask], points2_norm[inlier_mask]
        )
        
        # Combine inlier masks
        final_inliers = np.zeros(len(points1), dtype=bool)
        inlier_indices = np.where(inlier_mask)[0]
        if len(pose_inliers) > 0 and np.any(pose_inliers):
            pose_inlier_indices = inlier_indices[pose_inliers.astype(bool)]
            final_inliers[pose_inlier_indices] = True
        
        # Compute fundamental matrix
        F = self._essential_to_fundamental(E, K1, K2)
        
        positive_count = int(pose_inliers.astype(bool).sum())
        rotation_angle = self._rotation_angle_degrees(R)

        return {
            'R': R,
            't': t,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'inlier_mask': final_inliers,
            'num_inliers': np.sum(final_inliers),
            'inlier_ratio': np.sum(final_inliers) / len(points1),
            'success': True,
            'positive_depth_count': positive_count,
            'rotation_angle_deg': rotation_angle
        }
    
    def _estimate_pose_opencv(self,
                            points1: np.ndarray,
                            points2: np.ndarray,
                            K1: np.ndarray,
                            K2: np.ndarray) -> Dict:
        """Estimate pose using OpenCV's robust solvers."""
        # Estimate essential matrix
        E, inlier_mask = cv2.findEssentialMat(
            points1, points2, K1,
            method=cv2.RANSAC,
            prob=self.ransac_confidence,
            threshold=self.ransac_threshold,
            maxIters=self.ransac_max_iters
        )
        
        if E is None:
            return {'success': False}
        
        inlier_mask = inlier_mask.ravel().astype(bool)
        
        # Recover pose
        _, R, t, pose_mask = cv2.recoverPose(
            E, points1[inlier_mask], points2[inlier_mask], K1
        )
        
        # Combine masks
        final_inliers = np.zeros(len(points1), dtype=bool)
        inlier_indices = np.where(inlier_mask)[0]
        pose_inliers_bool = pose_mask.ravel().astype(bool)
        if len(pose_inliers_bool) > 0:
            valid_indices = inlier_indices[pose_inliers_bool]
            final_inliers[valid_indices] = True
        
        # Compute fundamental matrix
        F = self._essential_to_fundamental(E, K1, K2)

        positive_count = int(pose_mask.astype(bool).sum())
        rotation_angle = self._rotation_angle_degrees(R)
        
        return {
            'R': R,
            't': t,
            'essential_matrix': E,
            'fundamental_matrix': F,
            'inlier_mask': final_inliers,
            'num_inliers': np.sum(final_inliers),
            'inlier_ratio': np.sum(final_inliers) / len(points1),
            'success': True,
            'positive_depth_count': positive_count,
            'rotation_angle_deg': rotation_angle
        }
    
    def estimate_pose_pnp(self,
                         points_3d: np.ndarray,
                         points_2d: np.ndarray,
                         K: np.ndarray,
                         dist_coeffs: Optional[np.ndarray] = None) -> Dict:
        """
        Estimate camera pose using PnP (Perspective-n-Point) solver.
        
        Args:
            points_3d: [N, 3] 3D points in world coordinates
            points_2d: [N, 2] corresponding 2D points in image coordinates
            K: [3, 3] camera intrinsic matrix
            dist_coeffs: [4/5/8] distortion coefficients (optional)
            
        Returns:
            Dictionary with pose estimation results
        """
        result = {
            'R': None,
            't': None,
            'rvec': None,
            'tvec': None,
            'inlier_mask': None,
            'num_inliers': 0,
            'inlier_ratio': 0.0,
            'success': False
        }
        
        if len(points_3d) < 4:
            logging.warning(f"Insufficient 3D-2D correspondences: {len(points_3d)} < 4")
            return result
        
        if dist_coeffs is None:
            dist_coeffs = np.zeros(4)
        
        try:
            # Use RANSAC-based PnP solver
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                points_3d.astype(np.float32),
                points_2d.astype(np.float32),
                K.astype(np.float32),
                dist_coeffs.astype(np.float32),
                confidence=self.ransac_confidence,
                reprojectionError=self.ransac_threshold,
                iterationsCount=self.ransac_max_iters
            )
            
            if success and inliers is not None:
                # Convert rotation vector to rotation matrix
                R, _ = cv2.Rodrigues(rvec)
                
                # Create inlier mask
                inlier_mask = np.zeros(len(points_3d), dtype=bool)
                inlier_mask[inliers.ravel()] = True
                
                result.update({
                    'R': R,
                    't': tvec,
                    'rvec': rvec,
                    'tvec': tvec,
                    'inlier_mask': inlier_mask,
                    'num_inliers': len(inliers),
                    'inlier_ratio': len(inliers) / len(points_3d),
                    'success': True
                })
                
                logging.info(f"PnP estimation successful: {len(inliers)}/{len(points_3d)} inliers")
            
        except Exception as e:
            logging.error(f"PnP estimation failed: {e}")
        
        return result
    
    def _normalize_points(self, points: np.ndarray, K: np.ndarray) -> np.ndarray:
        """Convert pixel coordinates to normalized camera coordinates."""
        points_h = np.hstack([points, np.ones((len(points), 1))])
        K_inv = np.linalg.inv(K)
        points_norm = (K_inv @ points_h.T).T
        return points_norm[:, :2]
    
    def _recover_pose_from_essential(self,
                                   E: np.ndarray,
                                   points1_norm: np.ndarray,
                                   points2_norm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Recover pose from essential matrix using normalized coordinates."""
        # Decompose essential matrix
        U, S, Vt = np.linalg.svd(E)
        
        # Ensure proper rotation
        if np.linalg.det(U) < 0:
            U[:, -1] *= -1
        if np.linalg.det(Vt) < 0:
            Vt[-1, :] *= -1
        
        # Two possible rotations
        W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        R1 = U @ W @ Vt
        R2 = U @ W.T @ Vt
        
        # Translation (up to scale)
        t = U[:, 2].reshape(3, 1)
        
        # Test all four combinations
        poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]
        best_pose = None
        max_positive = 0
        
        for R, t_test in poses:
            # Triangulate points to test cheirality
            positive_count = self._count_positive_depth(
                R, t_test, points1_norm, points2_norm
            )
            
            if positive_count > max_positive:
                max_positive = positive_count
                best_pose = (R, t_test)
        
        if best_pose is None:
            # Fallback to first pose
            best_pose = poses[0]
        
        R, t = best_pose
        
        # Create inlier mask based on positive depth
        inlier_mask = self._get_positive_depth_mask(R, t, points1_norm, points2_norm)
        
        return R, t, inlier_mask
    
    def _count_positive_depth(self,
                            R: np.ndarray,
                            t: np.ndarray,
                            points1_norm: np.ndarray,
                            points2_norm: np.ndarray) -> int:
        """Count points with positive depth in both cameras."""
        # Simple triangulation to check depth
        count = 0
        
        # Camera matrices (normalized coordinates)
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        
        for p1, p2 in zip(points1_norm, points2_norm):
            # Triangulate point
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1]
            ])
            
            try:
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]  # Homogeneous to 3D
                
                # Check depth in both cameras
                depth1 = X[2]
                depth2 = (R @ X[:3].reshape(3, 1) + t)[2, 0]
                
                if depth1 > 0 and depth2 > 0:
                    count += 1
                    
            except:
                continue
        
        return count
    
    def _get_positive_depth_mask(self,
                               R: np.ndarray,
                               t: np.ndarray,
                               points1_norm: np.ndarray,
                               points2_norm: np.ndarray) -> np.ndarray:
        """Get mask of points with positive depth in both cameras."""
        mask = np.zeros(len(points1_norm), dtype=bool)
        
        # Camera matrices
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([R, t])
        
        for i, (p1, p2) in enumerate(zip(points1_norm, points2_norm)):
            try:
                # Triangulate point
                A = np.array([
                    p1[0] * P1[2] - P1[0],
                    p1[1] * P1[2] - P1[1],
                    p2[0] * P2[2] - P2[0],
                    p2[1] * P2[2] - P2[1]
                ])
                
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                X = X / X[3]
                
                # Check depth
                depth1 = X[2]
                depth2 = (R @ X[:3].reshape(3, 1) + t)[2, 0]
                
                if depth1 > 0 and depth2 > 0:
                    mask[i] = True
                    
            except:
                continue
        
        return mask

    def _estimate_parallax(self, points1: np.ndarray, points2: np.ndarray) -> Optional[float]:
        """Estimate parallax between two sets of pixel coordinates in degrees."""
        if len(points1) == 0 or len(points2) == 0:
            return None
        centroid1 = points1.mean(axis=0)
        centroid2 = points2.mean(axis=0)
        vectors1 = points1 - centroid1
        vectors2 = points2 - centroid2
        norms1 = np.linalg.norm(vectors1, axis=1)
        norms2 = np.linalg.norm(vectors2, axis=1)
        valid = (norms1 > 1e-6) & (norms2 > 1e-6)
        if not np.any(valid):
            return None
        vectors1 = vectors1[valid] / norms1[valid, None]
        vectors2 = vectors2[valid] / norms2[valid, None]
        dots = np.clip(np.sum(vectors1 * vectors2, axis=1), -1.0, 1.0)
        angles = np.degrees(np.arccos(dots))
        return float(np.median(angles)) if len(angles) else None

    @staticmethod
    def _rotation_angle_degrees(R: np.ndarray) -> float:
        """Compute absolute rotation angle of matrix R in degrees."""
        try:
            value = (np.trace(R) - 1.0) / 2.0
            value = np.clip(value, -1.0, 1.0)
            return float(np.degrees(np.arccos(value)))
        except Exception:
            return 0.0
    
    def _essential_to_fundamental(self,
                                E: np.ndarray,
                                K1: np.ndarray,
                                K2: np.ndarray) -> np.ndarray:
        """Convert essential matrix to fundamental matrix."""
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)
        F = K2_inv.T @ E @ K1_inv
        return F
    
    def validate_pose(self,
                     pose_result: Dict,
                     points1: np.ndarray,
                     points2: np.ndarray) -> bool:
        """
        Validate pose estimation result using geometric constraints.
        
        Args:
            pose_result: Result from pose estimation
            points1, points2: Original point correspondences
            
        Returns:
            True if pose passes validation checks
        """
        if not pose_result['success']:
            return False
        
        try:
            # Check minimum inlier ratio
            min_inlier_ratio = 0.1
            if pose_result['inlier_ratio'] < min_inlier_ratio:
                logging.warning(f"Low inlier ratio: {pose_result['inlier_ratio']:.3f}")
                return False
            
            # Check rotation matrix validity
            R = pose_result['R']
            if not self._is_valid_rotation_matrix(R):
                logging.warning("Invalid rotation matrix")
                return False
            
            # Check translation vector
            t = pose_result['t']
            if np.linalg.norm(t) < 1e-6:
                logging.warning("Translation vector too small")
                return False
            
            # Check epipolar constraint
            if 'fundamental_matrix' in pose_result and pose_result['fundamental_matrix'] is not None:
                F = pose_result['fundamental_matrix']
                inliers = pose_result['inlier_mask']
                
                epipolar_errors = self._compute_epipolar_errors(
                    points1[inliers], points2[inliers], F
                )
                median_error = np.median(epipolar_errors)
                
                if median_error > self.ransac_threshold * 2:
                    logging.warning(f"High epipolar error: {median_error:.3f}")
                    return False
            
            return True
            
        except Exception as e:
            logging.warning(f"Pose validation failed: {e}")
            return False
    
    def _is_valid_rotation_matrix(self, R: np.ndarray) -> bool:
        """Check if matrix is a valid rotation matrix."""
        if R.shape != (3, 3):
            return False
        
        # Check orthogonality
        should_be_identity = R @ R.T
        identity = np.eye(3)
        
        # Check determinant
        det = np.linalg.det(R)
        
        return (np.allclose(should_be_identity, identity, atol=1e-2) and 
                np.allclose(det, 1.0, atol=1e-2))
    
    def _compute_epipolar_errors(self,
                               points1: np.ndarray,
                               points2: np.ndarray,
                               F: np.ndarray) -> np.ndarray:
        """Compute symmetric epipolar errors."""
        # Convert to homogeneous coordinates
        points1_h = np.hstack([points1, np.ones((len(points1), 1))])
        points2_h = np.hstack([points2, np.ones((len(points2), 1))])
        
        # Compute epipolar lines
        lines1 = F.T @ points2_h.T  # Lines in image 1
        lines2 = F @ points1_h.T    # Lines in image 2
        
        # Point-to-line distances
        errors1 = np.abs(np.sum(points1_h * lines1.T, axis=1)) / np.sqrt(lines1[0]**2 + lines1[1]**2)
        errors2 = np.abs(np.sum(points2_h * lines2.T, axis=1)) / np.sqrt(lines2[0]**2 + lines2[1]**2)
        
        # Symmetric error
        return (errors1 + errors2) / 2


def create_pose_estimator(config: Dict) -> PoseEstimator:
    """Factory function to create pose estimator."""
    return PoseEstimator(config)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open("../../config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create pose estimator
    pose_estimator = create_pose_estimator(config)
    
    print("Pose Estimator initialized successfully!")
    print(f"Preferred method: {pose_estimator.preferred_method}")
    print(f"COLMAP available: {COLMAP_AVAILABLE}")
