"""
Classical Triangulation Module for Modern SfM Pipeline

This module implements robust 3D point triangulation using classical geometric
methods with comprehensive quality filtering and validation.

Author: ModernSFM Pipeline
Date: July 25th, 2025
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path


class Triangulator:
    """
    Robust 3D point triangulation using classical geometric methods.
    
    Provides multi-view triangulation with comprehensive quality filtering
    based on triangulation angles, reprojection errors, and geometric constraints.
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.triangulation_config = config.get('reconstruction', {})
        
        # Quality thresholds
        self.min_triangulation_angle = self.triangulation_config.get('min_triangulation_angle', 2.0)
        self.max_reprojection_error = self.triangulation_config.get('max_reprojection_error', 4.0)
        self.min_track_length = self.triangulation_config.get('min_track_length', 2)
        self.max_track_length = self.triangulation_config.get('max_track_length', 50)
        
        logging.info(f"Initialized Triangulator with:")
        logging.info(f"  Min triangulation angle: {self.min_triangulation_angle}°")
        logging.info(f"  Max reprojection error: {self.max_reprojection_error} pixels")
        logging.info(f"  Track length range: {self.min_track_length}-{self.max_track_length}")
    
    def triangulate_two_view(self,
                           points1: np.ndarray,
                           points2: np.ndarray,
                           P1: np.ndarray,
                           P2: np.ndarray,
                           method: str = 'dlt') -> Dict:
        """
        Triangulate 3D points from two views.
        
        Args:
            points1, points2: [N, 2] corresponding points in pixel coordinates
            P1, P2: [3, 4] camera projection matrices
            method: 'dlt', 'optimal', or 'opencv'
            
        Returns:
            Dictionary containing:
                - points_3d: [N, 3] triangulated 3D points
                - reprojection_errors: [N] reprojection errors for each point
                - triangulation_angles: [N] triangulation angles in degrees
                - valid_mask: [N] boolean mask of valid triangulations
                - method_used: str method that was used
        """
        result = {
            'points_3d': None,
            'reprojection_errors': None,
            'triangulation_angles': None,
            'valid_mask': None,
            'method_used': method
        }
        
        if len(points1) == 0:
            logging.warning("No correspondences provided for triangulation")
            return result
        
        # Choose triangulation method
        if method == 'opencv':
            points_3d = self._triangulate_opencv(points1, points2, P1, P2)
        elif method == 'optimal':
            points_3d = self._triangulate_optimal(points1, points2, P1, P2)
        else:  # DLT
            points_3d = self._triangulate_dlt(points1, points2, P1, P2)
        
        if points_3d is None:
            return result
        
        # Compute quality metrics
        reprojection_errors = self._compute_reprojection_errors(
            points_3d, points1, points2, P1, P2
        )
        
        triangulation_angles = self._compute_triangulation_angles(
            points_3d, P1, P2
        )
        
        # Filter based on quality criteria
        valid_mask = self._filter_triangulated_points(
            points_3d, reprojection_errors, triangulation_angles
        )
        
        result.update({
            'points_3d': points_3d,
            'reprojection_errors': reprojection_errors,
            'triangulation_angles': triangulation_angles,
            'valid_mask': valid_mask
        })
        
        num_valid = np.sum(valid_mask) if valid_mask is not None else 0
        logging.info(f"Triangulated {num_valid}/{len(points1)} valid points using {method}")
        
        return result
    
    def triangulate_multi_view(self,
                             observations: Dict[int, np.ndarray],
                             camera_matrices: Dict[int, np.ndarray],
                             method: str = 'dlt') -> Dict:
        """
        Triangulate 3D points from multiple views.
        
        Args:
            observations: Dict mapping camera_id -> [N, 2] observed points
            camera_matrices: Dict mapping camera_id -> [3, 4] projection matrix
            method: Triangulation method to use
            
        Returns:
            Dictionary with triangulation results
        """
        if len(observations) < 2:
            logging.warning("Need at least 2 views for triangulation")
            return {'points_3d': None, 'valid_mask': None}
        
        # Get common point indices across all views
        camera_ids = list(observations.keys())
        num_points = len(observations[camera_ids[0]])
        
        # Verify all views have same number of points
        for cam_id in camera_ids:
            if len(observations[cam_id]) != num_points:
                logging.error("Inconsistent number of points across views")
                return {'points_3d': None, 'valid_mask': None}
        
        points_3d = np.zeros((num_points, 3))
        reprojection_errors = np.zeros(num_points)
        triangulation_angles = np.zeros(num_points)
        valid_mask = np.zeros(num_points, dtype=bool)
        
        # Triangulate each point
        for i in range(num_points):
            # Collect observations for this point
            point_obs = {}
            point_cameras = {}
            
            for cam_id in camera_ids:
                point_obs[cam_id] = observations[cam_id][i]
                point_cameras[cam_id] = camera_matrices[cam_id]
            
            # Triangulate using all available views
            point_result = self._triangulate_point_multi_view(
                point_obs, point_cameras, method
            )
            
            if point_result['success']:
                points_3d[i] = point_result['point_3d']
                reprojection_errors[i] = point_result['reprojection_error']
                triangulation_angles[i] = point_result['triangulation_angle']
                valid_mask[i] = point_result['valid']
        
        num_valid = np.sum(valid_mask)
        logging.info(f"Multi-view triangulation: {num_valid}/{num_points} valid points")
        
        return {
            'points_3d': points_3d,
            'reprojection_errors': reprojection_errors,
            'triangulation_angles': triangulation_angles,
            'valid_mask': valid_mask,
            'method_used': method
        }
    
    def _triangulate_dlt(self,
                        points1: np.ndarray,
                        points2: np.ndarray,
                        P1: np.ndarray,
                        P2: np.ndarray) -> np.ndarray:
        """Triangulate using Direct Linear Transform (DLT)."""
        points_3d = np.zeros((len(points1), 3))
        
        for i, (p1, p2) in enumerate(zip(points1, points2)):
            # Build the system Ax = 0
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1]
            ])
            
            try:
                # Solve using SVD
                _, _, Vt = np.linalg.svd(A)
                X = Vt[-1]
                
                # Convert from homogeneous to 3D
                if abs(X[3]) > 1e-10:
                    points_3d[i] = X[:3] / X[3]
                else:
                    points_3d[i] = X[:3]  # Point at infinity
                    
            except np.linalg.LinAlgError:
                points_3d[i] = np.array([0, 0, 0])
        
        return points_3d
    
    def _triangulate_optimal(self,
                           points1: np.ndarray,
                           points2: np.ndarray,
                           P1: np.ndarray,
                           P2: np.ndarray) -> np.ndarray:
        """Triangulate using optimal method (Hartley & Zisserman)."""
        points_3d = np.zeros((len(points1), 3))
        
        for i, (p1, p2) in enumerate(zip(points1, points2)):
            # Initial estimate using DLT
            A = np.array([
                p1[0] * P1[2] - P1[0],
                p1[1] * P1[2] - P1[1],
                p2[0] * P2[2] - P2[0],
                p2[1] * P2[2] - P2[1]
            ])
            
            try:
                _, _, Vt = np.linalg.svd(A)
                X_init = Vt[-1]
                
                if abs(X_init[3]) > 1e-10:
                    X_init = X_init / X_init[3]
                
                # Refine using iterative optimization
                X_optimal = self._refine_triangulation(p1, p2, P1, P2, X_init[:3])
                points_3d[i] = X_optimal
                
            except (np.linalg.LinAlgError, ValueError):
                points_3d[i] = np.array([0, 0, 0])
        
        return points_3d
    
    def _triangulate_opencv(self,
                          points1: np.ndarray,
                          points2: np.ndarray,
                          P1: np.ndarray,
                          P2: np.ndarray) -> np.ndarray:
        """Triangulate using OpenCV's triangulatePoints."""
        # OpenCV expects points in homogeneous coordinates
        points1_h = points1.T
        points2_h = points2.T
        
        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, points1_h, points2_h)
        
        # Convert to 3D
        points_3d = points_4d[:3] / points_4d[3]
        
        return points_3d.T
    
    def _triangulate_point_multi_view(self,
                                    observations: Dict[int, np.ndarray],
                                    cameras: Dict[int, np.ndarray],
                                    method: str) -> Dict:
        """Triangulate a single point from multiple views."""
        camera_ids = list(observations.keys())
        
        if len(camera_ids) < 2:
            return {'success': False}
        
        # Build overdetermined system for DLT
        equations = []
        for cam_id in camera_ids:
            p = observations[cam_id]
            P = cameras[cam_id]
            
            equations.append(p[0] * P[2] - P[0])
            equations.append(p[1] * P[2] - P[1])
        
        A = np.array(equations)
        
        try:
            # Solve using SVD
            _, _, Vt = np.linalg.svd(A)
            X = Vt[-1]
            
            # Convert to 3D
            if abs(X[3]) > 1e-10:
                point_3d = X[:3] / X[3]
            else:
                point_3d = X[:3]
            
            # Compute quality metrics
            reprojection_error = self._compute_multi_view_reprojection_error(
                point_3d, observations, cameras
            )
            
            triangulation_angle = self._compute_multi_view_triangulation_angle(
                point_3d, cameras
            )
            
            # Check validity
            valid = (reprojection_error < self.max_reprojection_error and
                    triangulation_angle > self.min_triangulation_angle)
            
            return {
                'success': True,
                'point_3d': point_3d,
                'reprojection_error': reprojection_error,
                'triangulation_angle': triangulation_angle,
                'valid': valid
            }
            
        except np.linalg.LinAlgError:
            return {'success': False}
    
    def _refine_triangulation(self,
                            p1: np.ndarray,
                            p2: np.ndarray,
                            P1: np.ndarray,
                            P2: np.ndarray,
                            X_init: np.ndarray,
                            max_iterations: int = 10) -> np.ndarray:
        """Refine triangulation using iterative optimization."""
        X = X_init.copy()
        
        for _ in range(max_iterations):
            # Project to both cameras
            x1_proj = P1 @ np.append(X, 1)
            x2_proj = P2 @ np.append(X, 1)
            
            x1_proj = x1_proj[:2] / x1_proj[2]
            x2_proj = x2_proj[:2] / x2_proj[2]
            
            # Compute residuals
            r1 = p1 - x1_proj
            r2 = p2 - x2_proj
            
            # Check convergence
            if np.linalg.norm(r1) + np.linalg.norm(r2) < 1e-6:
                break
            
            # Compute Jacobian
            J = self._compute_triangulation_jacobian(X, P1, P2)
            
            # Update using Gauss-Newton
            residual = np.concatenate([r1, r2])
            
            try:
                delta = np.linalg.lstsq(J, residual, rcond=None)[0]
                X += delta
            except np.linalg.LinAlgError:
                break
        
        return X
    
    def _compute_triangulation_jacobian(self,
                                      X: np.ndarray,
                                      P1: np.ndarray,
                                      P2: np.ndarray) -> np.ndarray:
        """Compute Jacobian for triangulation refinement."""
        # Homogeneous 3D point
        X_h = np.append(X, 1)
        
        # Project to cameras
        x1_h = P1 @ X_h
        x2_h = P2 @ X_h
        
        # Jacobian of projection
        J1 = np.array([
            [P1[0, 0] * x1_h[2] - P1[2, 0] * x1_h[0],
             P1[0, 1] * x1_h[2] - P1[2, 1] * x1_h[0],
             P1[0, 2] * x1_h[2] - P1[2, 2] * x1_h[0]],
            [P1[1, 0] * x1_h[2] - P1[2, 0] * x1_h[1],
             P1[1, 1] * x1_h[2] - P1[2, 1] * x1_h[1],
             P1[1, 2] * x1_h[2] - P1[2, 2] * x1_h[1]]
        ]) / (x1_h[2] ** 2)
        
        J2 = np.array([
            [P2[0, 0] * x2_h[2] - P2[2, 0] * x2_h[0],
             P2[0, 1] * x2_h[2] - P2[2, 1] * x2_h[0],
             P2[0, 2] * x2_h[2] - P2[2, 2] * x2_h[0]],
            [P2[1, 0] * x2_h[2] - P2[2, 0] * x2_h[1],
             P2[1, 1] * x2_h[2] - P2[2, 1] * x2_h[1],
             P2[1, 2] * x2_h[2] - P2[2, 2] * x2_h[1]]
        ]) / (x2_h[2] ** 2)
        
        return np.vstack([J1, J2])
    
    def _compute_reprojection_errors(self,
                                   points_3d: np.ndarray,
                                   points1: np.ndarray,
                                   points2: np.ndarray,
                                   P1: np.ndarray,
                                   P2: np.ndarray) -> np.ndarray:
        """Compute reprojection errors for triangulated points."""
        errors = np.zeros(len(points_3d))
        
        for i, (X, p1, p2) in enumerate(zip(points_3d, points1, points2)):
            # Project 3D point to both cameras
            X_h = np.append(X, 1)
            
            x1_proj = P1 @ X_h
            x2_proj = P2 @ X_h
            
            # Convert to pixel coordinates
            if abs(x1_proj[2]) > 1e-10:
                x1_proj = x1_proj[:2] / x1_proj[2]
            else:
                x1_proj = x1_proj[:2]
            
            if abs(x2_proj[2]) > 1e-10:
                x2_proj = x2_proj[:2] / x2_proj[2]
            else:
                x2_proj = x2_proj[:2]
            
            # Compute symmetric reprojection error
            error1 = np.linalg.norm(p1 - x1_proj)
            error2 = np.linalg.norm(p2 - x2_proj)
            errors[i] = (error1 + error2) / 2
        
        return errors
    
    def _compute_multi_view_reprojection_error(self,
                                             point_3d: np.ndarray,
                                             observations: Dict[int, np.ndarray],
                                             cameras: Dict[int, np.ndarray]) -> float:
        """Compute reprojection error across multiple views."""
        total_error = 0.0
        num_views = 0
        
        X_h = np.append(point_3d, 1)
        
        for cam_id in observations.keys():
            p_obs = observations[cam_id]
            P = cameras[cam_id]
            
            # Project to camera
            x_proj = P @ X_h
            
            if abs(x_proj[2]) > 1e-10:
                x_proj = x_proj[:2] / x_proj[2]
                error = np.linalg.norm(p_obs - x_proj)
                total_error += error
                num_views += 1
        
        return total_error / max(num_views, 1)
    
    def _compute_triangulation_angles(self,
                                    points_3d: np.ndarray,
                                    P1: np.ndarray,
                                    P2: np.ndarray) -> np.ndarray:
        """Compute triangulation angles for quality assessment."""
        angles = np.zeros(len(points_3d))
        
        # Extract camera centers
        C1 = self._get_camera_center(P1)
        C2 = self._get_camera_center(P2)
        
        for i, X in enumerate(points_3d):
            # Vectors from cameras to 3D point
            v1 = X - C1
            v2 = X - C2
            
            # Compute angle between rays
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Handle numerical errors
            
            angle = np.arccos(cos_angle)
            angles[i] = np.degrees(angle)
        
        return angles
    
    def _compute_multi_view_triangulation_angle(self,
                                              point_3d: np.ndarray,
                                              cameras: Dict[int, np.ndarray]) -> float:
        """Compute minimum triangulation angle across all camera pairs."""
        camera_ids = list(cameras.keys())
        min_angle = 180.0
        
        # Check all pairs of cameras
        for i in range(len(camera_ids)):
            for j in range(i + 1, len(camera_ids)):
                C1 = self._get_camera_center(cameras[camera_ids[i]])
                C2 = self._get_camera_center(cameras[camera_ids[j]])
                
                # Vectors from cameras to 3D point
                v1 = point_3d - C1
                v2 = point_3d - C2
                
                # Compute angle
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                
                angle = np.degrees(np.arccos(cos_angle))
                min_angle = min(min_angle, angle)
        
        return min_angle
    
    def _get_camera_center(self, P: np.ndarray) -> np.ndarray:
        """Extract camera center from projection matrix."""
        # Camera center is the null space of P
        # P = K[R|t], C = -R^T * t
        try:
            _, _, Vt = np.linalg.svd(P)
            C_h = Vt[-1]
            
            if abs(C_h[3]) > 1e-10:
                return C_h[:3] / C_h[3]
            else:
                return C_h[:3]
        except np.linalg.LinAlgError:
            return np.array([0, 0, 0])
    
    def _filter_triangulated_points(self,
                                  points_3d: np.ndarray,
                                  reprojection_errors: np.ndarray,
                                  triangulation_angles: np.ndarray) -> np.ndarray:
        """Filter triangulated points based on quality criteria."""
        valid_mask = np.ones(len(points_3d), dtype=bool)
        
        # Filter by reprojection error
        valid_mask &= reprojection_errors < self.max_reprojection_error
        
        # Filter by triangulation angle
        valid_mask &= triangulation_angles > self.min_triangulation_angle
        
        # Filter points at infinity or behind cameras
        valid_mask &= np.all(np.isfinite(points_3d), axis=1)
        
        # Filter points too far from origin (likely outliers)
        distances = np.linalg.norm(points_3d, axis=1)
        median_distance = np.median(distances[valid_mask]) if np.any(valid_mask) else 1.0
        max_distance = median_distance * 100  # Allow up to 100x median distance
        valid_mask &= distances < max_distance
        
        return valid_mask
    
    def create_tracks(self,
                     matches_dict: Dict[Tuple[int, int], np.ndarray],
                     num_images: int) -> Dict[int, Dict[int, int]]:
        """
        Create point tracks from pairwise matches.
        
        Args:
            matches_dict: Dict mapping (img1_id, img2_id) -> matches array
            num_images: Total number of images
            
        Returns:
            Dict mapping track_id -> {image_id: point_index}
        """
        # Build correspondence graph
        correspondences = {}  # (img_id, point_idx) -> set of (other_img_id, other_point_idx)
        
        for (img1, img2), matches in matches_dict.items():
            for match in matches:
                pt1_idx, pt2_idx = int(match[0]), int(match[1])
                
                key1 = (img1, pt1_idx)
                key2 = (img2, pt2_idx)
                
                if key1 not in correspondences:
                    correspondences[key1] = set()
                if key2 not in correspondences:
                    correspondences[key2] = set()
                
                correspondences[key1].add(key2)
                correspondences[key2].add(key1)
        
        # Find connected components (tracks)
        visited = set()
        tracks = {}
        track_id = 0
        
        for key in correspondences:
            if key in visited:
                continue
            
            # BFS to find connected component
            track = {}
            queue = [key]
            visited.add(key)
            
            while queue:
                current = queue.pop(0)
                img_id, pt_idx = current
                track[img_id] = pt_idx
                
                for neighbor in correspondences[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            # Only keep tracks with sufficient length
            if len(track) >= self.min_track_length:
                tracks[track_id] = track
                track_id += 1
        
        logging.info(f"Created {len(tracks)} tracks from {len(correspondences)} correspondences")
        
        return tracks


def create_triangulator(config: Dict) -> Triangulator:
    """Factory function to create triangulator."""
    return Triangulator(config)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open("../../config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create triangulator
    triangulator = create_triangulator(config)
    
    print("Triangulator initialized successfully!")
    print(f"Min triangulation angle: {triangulator.min_triangulation_angle}°")
    print(f"Max reprojection error: {triangulator.max_reprojection_error} pixels")
