"""
Evaluation metrics for comparing SfM reconstruction results against ground truth.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging
import json
import re
from scipy.spatial.transform import Rotation as ScipyRotation
from scipy.spatial.distance import cdist

from .ground_truth import GroundTruthData, load_ground_truth

logger = logging.getLogger(__name__)


class SyntheticEvaluator:
    """Evaluate SfM reconstruction results against synthetic ground truth."""

    def __init__(self, ground_truth_path: Path):
        """Initialize evaluator with ground truth data."""
        self.ground_truth = load_ground_truth(ground_truth_path)
        logger.info("Synthetic evaluator initialized")

    def evaluate_reconstruction(self, reconstruction_dir: Path) -> Dict[str, Any]:
        """
        Evaluate complete reconstruction against ground truth.

        Args:
            reconstruction_dir: Directory containing reconstruction results

        Returns:
            evaluation_results: Comprehensive evaluation metrics
        """
        logger.info(f"Evaluating reconstruction in {reconstruction_dir}")

        results = {
            'timestamp': self.ground_truth.timestamp,
            'ground_truth_scene': self.ground_truth.scene.scene_type,
            'evaluation_summary': {}
        }

        try:
            # Load reconstruction results
            reconstruction_data = self._load_reconstruction_results(reconstruction_dir)

            alignment_info = self._align_reconstruction_to_ground_truth(reconstruction_data)
            if alignment_info:
                results['alignment'] = alignment_info

            # Evaluate each component
            if 'camera_poses' in reconstruction_data:
                pose_metrics = self.evaluate_pose_estimation(reconstruction_data['camera_poses'])
                results['pose_estimation'] = pose_metrics
                results['evaluation_summary']['pose_estimation_passed'] = pose_metrics['overall']['mean_rotation_error'] < 5.0

            if 'points_3d' in reconstruction_data:
                point_metrics = self.evaluate_triangulation(reconstruction_data['points_3d'])
                results['triangulation'] = point_metrics
                results['evaluation_summary']['triangulation_passed'] = point_metrics['overall']['mean_distance_error'] < 0.1

            if 'bundle_adjustment' in reconstruction_data:
                ba_metrics = self.evaluate_bundle_adjustment(reconstruction_data['bundle_adjustment'])
                results['bundle_adjustment'] = ba_metrics
                results['evaluation_summary']['bundle_adjustment_improved'] = ba_metrics['convergence']['error_reduction'] > 0.1

            # Overall assessment
            results['evaluation_summary']['overall_success'] = self._assess_overall_quality(results)

            logger.info("Evaluation completed successfully")
            return results

        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            results['error'] = str(e)
            return results

    def evaluate_pose_estimation(self, estimated_poses: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate camera pose estimation accuracy."""
        logger.info("Evaluating pose estimation accuracy")

        results = {
            'method': 'pose_comparison',
            'n_cameras': len(estimated_poses),
            'per_camera': {},
            'overall': {}
        }

        rotation_errors = []
        translation_errors = []

        for camera_name, pose_data in estimated_poses.items():
            if camera_name not in self.ground_truth.cameras:
                logger.warning(f"Camera {camera_name} not in ground truth, skipping")
                continue

            # Get ground truth pose
            gt_camera = self.ground_truth.cameras[camera_name]
            gt_R = gt_camera.rotation
            gt_t = gt_camera.translation

            # Get estimated pose
            if 'rotation_matrix' in pose_data:
                est_R = np.array(pose_data['rotation_matrix'])
                est_t = np.array(pose_data['translation'])
            else:
                # Convert from other formats if needed
                logger.warning(f"Unknown pose format for {camera_name}")
                continue

            # Calculate errors
            rot_error = self._rotation_error(gt_R, est_R)
            trans_error = self._translation_error(gt_t, est_t)

            rotation_errors.append(rot_error)
            translation_errors.append(trans_error)

            results['per_camera'][camera_name] = {
                'rotation_error_degrees': float(rot_error),
                'translation_error': float(trans_error)
            }

        # Overall statistics
        if rotation_errors:
            results['overall'] = {
                'mean_rotation_error': float(np.mean(rotation_errors)),
                'std_rotation_error': float(np.std(rotation_errors)),
                'max_rotation_error': float(np.max(rotation_errors)),
                'mean_translation_error': float(np.mean(translation_errors)),
                'std_translation_error': float(np.std(translation_errors)),
                'max_translation_error': float(np.max(translation_errors)),
                'cameras_evaluated': len(rotation_errors)
            }

            logger.info(f"Pose evaluation: {results['overall']['mean_rotation_error']:.2f}° rotation, "
                       f"{results['overall']['mean_translation_error']:.4f} translation error")

        return results

    def evaluate_triangulation(self, reconstructed_points: np.ndarray) -> Dict[str, Any]:
        """Evaluate 3D point triangulation accuracy."""
        logger.info("Evaluating triangulation accuracy")

        results = {
            'method': 'point_comparison',
            'n_gt_points': len(self.ground_truth.scene.points_3d),
            'n_reconstructed_points': len(reconstructed_points),
            'per_point': {},
            'overall': {}
        }

        # Align reconstructed points to ground truth
        aligned_points, alignment_transform = self._align_point_clouds(
            reconstructed_points, self.ground_truth.scene.points_3d
        )

        # Find correspondences between reconstructed and ground truth points
        correspondences = self._find_point_correspondences(aligned_points, self.ground_truth.scene.points_3d)

        distance_errors = []
        matched_count = 0

        for gt_idx, recon_idx in correspondences:
            if recon_idx is not None:
                gt_point = self.ground_truth.scene.points_3d[gt_idx]
                recon_point = aligned_points[recon_idx]
                error = np.linalg.norm(gt_point - recon_point)

                distance_errors.append(error)
                matched_count += 1

                results['per_point'][f'gt_point_{gt_idx}'] = {
                    'matched': True,
                    'distance_error': float(error)
                }
            else:
                results['per_point'][f'gt_point_{gt_idx}'] = {
                    'matched': False,
                    'distance_error': float('inf')
                }

        # Overall statistics
        if distance_errors:
            results['overall'] = {
                'mean_distance_error': float(np.mean(distance_errors)),
                'std_distance_error': float(np.std(distance_errors)),
                'max_distance_error': float(np.max(distance_errors)),
                'min_distance_error': float(np.min(distance_errors)),
                'matched_points': matched_count,
                'match_ratio': float(matched_count / len(self.ground_truth.scene.points_3d)),
                'completeness': float(matched_count / len(self.ground_truth.scene.points_3d)),
                'precision': float(matched_count / len(reconstructed_points)) if len(reconstructed_points) > 0 else 0.0
            }

            logger.info(f"Triangulation evaluation: {results['overall']['mean_distance_error']:.4f} mean error, "
                       f"{results['overall']['match_ratio']:.2f} match ratio")

        return results

    def evaluate_bundle_adjustment(self, ba_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate bundle adjustment convergence and improvement."""
        logger.info("Evaluating bundle adjustment performance")

        results = {
            'method': 'bundle_adjustment_analysis',
            'convergence': {},
            'quality': {}
        }

        # Analyze convergence
        if 'convergence_history' in ba_results:
            history = ba_results['convergence_history']
            initial_error = history[0] if history else 0
            final_error = history[-1] if history else 0

            results['convergence'] = {
                'iterations': len(history),
                'initial_error': float(initial_error),
                'final_error': float(final_error),
                'error_reduction': float(initial_error - final_error) if initial_error > 0 else 0.0,
                'converged': ba_results.get('converged', False)
            }

        # Evaluate final poses and points if available
        if 'final_poses' in ba_results:
            pose_metrics = self.evaluate_pose_estimation(ba_results['final_poses'])
            results['quality']['final_pose_accuracy'] = pose_metrics['overall']

        if 'final_points' in ba_results:
            point_metrics = self.evaluate_triangulation(ba_results['final_points'])
            results['quality']['final_point_accuracy'] = point_metrics['overall']

        logger.info("Bundle adjustment evaluation completed")
        return results

    def _rotation_error(self, R_gt: np.ndarray, R_est: np.ndarray) -> float:
        """Calculate rotation error in degrees."""
        # Compute relative rotation
        R_rel = R_gt.T @ R_est

        # Convert to angle-axis and get angle
        r = ScipyRotation.from_matrix(R_rel)
        angle_rad = r.magnitude()

        # Convert to degrees
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    def _translation_error(self, t_gt: np.ndarray, t_est: np.ndarray) -> float:
        """Calculate translation error as Euclidean distance."""
        return np.linalg.norm(t_gt - t_est)

    @staticmethod
    def _camera_center_from_pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
        """Compute camera center from world-to-camera rotation and translation."""
        return -rotation.T @ translation.reshape(3)

    def _align_reconstruction_to_ground_truth(self, reconstruction_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Align reconstructed poses and points to ground truth world coordinates."""
        if 'camera_poses' not in reconstruction_data or not reconstruction_data['camera_poses']:
            logger.info("No camera poses available for alignment")
            return None

        # Ensure we have enough cameras in common to compute a similarity transform
        estimated_poses = reconstruction_data['camera_poses']
        gt_cameras = self.ground_truth.cameras

        common_names = sorted(set(estimated_poses.keys()) & set(gt_cameras.keys()))
        if len(common_names) < 3:
            logger.warning("Not enough common cameras (%d) for pose alignment, skipping", len(common_names))
            return None

        est_centers = []
        gt_centers = []
        for name in common_names:
            est_pose = estimated_poses[name]
            R_est = np.asarray(est_pose['rotation_matrix'])
            t_est = np.asarray(est_pose['translation'])
            est_centers.append(self._camera_center_from_pose(R_est, t_est))

            gt_camera = gt_cameras[name]
            R_gt = gt_camera.rotation
            t_gt = gt_camera.translation
            gt_centers.append(self._camera_center_from_pose(R_gt, t_gt))

        est_centers = np.asarray(est_centers)
        gt_centers = np.asarray(gt_centers)

        scale, rotation, translation = self._umeyama_alignment(est_centers, gt_centers)
        logger.info(
            "Applied similarity alignment: scale=%.6f, rotation_det=%.6f",
            scale,
            np.linalg.det(rotation)
        )

        # Align camera poses
        aligned_poses: Dict[str, Dict[str, List[float]]] = {}
        for name, pose in estimated_poses.items():
            R_est = np.asarray(pose['rotation_matrix'])
            t_est = np.asarray(pose['translation'])

            C_est = self._camera_center_from_pose(R_est, t_est)
            C_aligned = scale * (rotation @ C_est) + translation

            R_aligned = R_est @ rotation.T
            # Re-orthonormalize to avoid drift from numerical errors
            U, _, Vt = np.linalg.svd(R_aligned)
            R_aligned = U @ Vt

            t_aligned = -R_aligned @ C_aligned

            aligned_poses[name] = {
                'rotation_matrix': R_aligned.tolist(),
                'translation': t_aligned.tolist()
            }

        reconstruction_data['camera_poses'] = aligned_poses

        # Align reconstructed point cloud if available
        if 'points_3d' in reconstruction_data and reconstruction_data['points_3d'] is not None:
            points = reconstruction_data['points_3d']
            try:
                aligned_points = scale * (points @ rotation.T) + translation
                reconstruction_data['points_3d'] = aligned_points
            except Exception as point_error:
                logger.warning("Failed to align reconstructed points: %s", point_error)

        return {
            'scale': float(scale),
            'rotation_matrix': rotation.tolist(),
            'translation': translation.tolist(),
            'reference_cameras': common_names
        }

    @staticmethod
    def _umeyama_alignment(source: np.ndarray, target: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute similarity transform (scale, rotation, translation) aligning source to target.
        """
        if source.shape != target.shape:
            raise ValueError("Source and target must have the same shape for alignment")

        n_points = source.shape[0]
        if n_points < 3:
            raise ValueError("At least 3 points required for similarity alignment")

        mean_source = source.mean(axis=0)
        mean_target = target.mean(axis=0)

        source_centered = source - mean_source
        target_centered = target - mean_target

        covariance = (target_centered.T @ source_centered) / n_points

        U, D, Vt = np.linalg.svd(covariance)
        S = np.eye(3)
        if np.linalg.det(U @ Vt) < 0:
            S[-1, -1] = -1

        rotation = U @ S @ Vt

        var_source = np.sum(source_centered ** 2) / n_points
        if var_source < 1e-12:
            raise ValueError("Degenerate configuration for similarity alignment")

        scale = np.sum(D * np.diag(S)) / var_source
        translation = mean_target - scale * rotation @ mean_source

        return scale, rotation, translation

    def _align_point_clouds(self, points1: np.ndarray, points2: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Align two point clouds using Procrustes analysis."""
        try:
            from scipy.spatial.distance import cdist

            # Find rough correspondences for alignment
            distances = cdist(points1, points2)
            correspondences = []

            # Simple nearest neighbor matching for alignment
            used_gt = set()
            for i in range(min(len(points1), len(points2))):
                # Find closest unused ground truth point
                min_dist = float('inf')
                best_j = None

                for j in range(len(points2)):
                    if j not in used_gt and distances[i, j] < min_dist:
                        min_dist = distances[i, j]
                        best_j = j

                if best_j is not None:
                    correspondences.append((i, best_j))
                    used_gt.add(best_j)

            if len(correspondences) < 3:
                logger.warning("Not enough correspondences for alignment, using identity")
                return points1, {'method': 'identity'}

            # Extract corresponding points
            p1_matched = np.array([points1[i] for i, j in correspondences])
            p2_matched = np.array([points2[j] for i, j in correspondences])

            # Compute centroids
            centroid1 = np.mean(p1_matched, axis=0)
            centroid2 = np.mean(p2_matched, axis=0)

            # Center the points
            p1_centered = p1_matched - centroid1
            p2_centered = p2_matched - centroid2

            # Compute optimal rotation using SVD
            H = p1_centered.T @ p2_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation matrix
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Compute scale (optional - for now assume scale = 1)
            scale = 1.0

            # Compute translation
            t = centroid2 - scale * R @ centroid1

            # Apply transformation
            aligned_points = scale * (R @ points1.T).T + t

            transform = {
                'method': 'procrustes',
                'rotation': R.tolist(),
                'translation': t.tolist(),
                'scale': scale,
                'correspondences_used': len(correspondences)
            }

            logger.info(f"Aligned point clouds using {len(correspondences)} correspondences")
            return aligned_points, transform

        except Exception as e:
            logger.warning(f"Point cloud alignment failed: {e}, using identity")
            return points1, {'method': 'identity', 'error': str(e)}

    def _find_point_correspondences(self, points1: np.ndarray, points2: np.ndarray,
                                  threshold: float = 0.1) -> List[Tuple[int, Optional[int]]]:
        """Find correspondences between two point clouds."""
        distances = cdist(points2, points1)  # gt x reconstructed

        correspondences = []
        used_recon = set()

        for gt_idx in range(len(points2)):
            # Find closest reconstructed point
            min_dist = float('inf')
            best_recon_idx = None

            for recon_idx in range(len(points1)):
                if recon_idx not in used_recon and distances[gt_idx, recon_idx] < min_dist:
                    min_dist = distances[gt_idx, recon_idx]
                    best_recon_idx = recon_idx

            # Accept correspondence if within threshold
            if best_recon_idx is not None and min_dist < threshold:
                correspondences.append((gt_idx, best_recon_idx))
                used_recon.add(best_recon_idx)
            else:
                correspondences.append((gt_idx, None))

        return correspondences

    def _load_reconstruction_results(self, reconstruction_dir: Path) -> Dict[str, Any]:
        """Load reconstruction results from output directory."""
        results = {}

        try:
            # Look for common result files in both root and data/ subdirectory
            search_roots = [reconstruction_dir]
            data_dir = reconstruction_dir / "data"
            if data_dir.exists():
                search_roots.append(data_dir)

            result_files = [
                'reconstruction_report.json',
                'camera_poses.json',
                'points_3d.npy',
                'bundle_adjustment_results.json'
            ]

            found_files = {}
            missing_files = []

            for filename in result_files:
                filepath = None
                for root in search_roots:
                    candidate = root / filename
                    if candidate.exists():
                        filepath = candidate
                        break

                if filepath is None:
                    missing_files.append(filename)
                    continue

                try:
                    if filename.endswith('.json'):
                        with open(filepath, 'r') as f:
                            data = json.load(f)
                            key = filename.replace('.json', '').replace('_results', '')
                            if key == 'camera_poses':
                                data = self._normalize_camera_poses(data)
                            results[key] = data
                    elif filename.endswith('.npy'):
                        data = np.load(filepath)
                        key = filename.replace('.npy', '')
                        results[key] = data

                    found_files[filename] = filepath
                except Exception as load_error:
                    logger.error(f"Failed to load {filename} from {filepath}: {load_error}")

            if found_files:
                logger.info(
                    "Loaded reconstruction results: %s",
                    {name: str(path) for name, path in found_files.items()}
                )
            else:
                logger.warning(
                    "No reconstruction result files were loaded from %s",
                    [str(root) for root in search_roots]
                )

            if missing_files:
                logger.warning(
                    "Missing expected reconstruction files: %s. "
                    "Checked locations: %s",
                    missing_files,
                    [str(root) for root in search_roots]
                )

            return results

        except Exception as e:
            logger.error(f"Failed to load reconstruction results: {e}")
            return {}

    def _normalize_camera_poses(self, poses_data: Any) -> Dict[str, Dict[str, List[float]]]:
        """
        Normalize camera pose data to map camera names to rotation/translation.

        Accepts either a list of pose dicts with numeric camera_ids or a dict
        keyed by camera name, ensuring each entry exposes rotation_matrix and translation.
        """
        if poses_data is None:
            return {}

        normalized: Dict[str, Dict[str, List[float]]] = {}

        def convert_entry(camera_name: str, pose_entry: Dict[str, Any]) -> Optional[Dict[str, List[float]]]:
            rotation = pose_entry.get('rotation_matrix', pose_entry.get('R'))
            translation = pose_entry.get('translation', pose_entry.get('t'))

            if rotation is None or translation is None:
                logger.warning(
                    "Pose entry for %s missing rotation/translation keys: %s",
                    camera_name,
                    list(pose_entry.keys())
                )
                return None

            rotation_array = np.asarray(rotation)
            translation_array = np.asarray(translation).reshape(-1)

            if rotation_array.shape != (3, 3):
                logger.warning(
                    "Pose entry for %s has invalid rotation shape %s",
                    camera_name,
                    rotation_array.shape
                )
                return None

            if translation_array.shape[0] != 3:
                logger.warning(
                    "Pose entry for %s has invalid translation shape %s",
                    camera_name,
                    translation_array.shape
                )
                return None

            return {
                'rotation_matrix': rotation_array.tolist(),
                'translation': translation_array.tolist()
            }

        # If already a dict keyed by camera name, coerce values and return
        if isinstance(poses_data, dict):
            for camera_name, pose_entry in poses_data.items():
                if not isinstance(pose_entry, dict):
                    logger.warning(
                        "Pose entry for %s is not a dict (type=%s), skipping",
                        camera_name,
                        type(pose_entry)
                    )
                    continue

                converted = convert_entry(str(camera_name), pose_entry)
                if converted:
                    normalized[str(camera_name)] = converted
            return normalized

        # Otherwise expect a list of pose dicts with camera identifiers
        if isinstance(poses_data, list):
            id_lookup = self._build_camera_id_lookup()

            for pose_entry in poses_data:
                if not isinstance(pose_entry, dict):
                    logger.warning("Pose list entry is not a dict (type=%s), skipping", type(pose_entry))
                    continue

                camera_name: Optional[str] = None

                camera_id = pose_entry.get('camera_id')
                if camera_id is not None and camera_id in id_lookup:
                    camera_name = id_lookup[camera_id]
                elif 'camera_name' in pose_entry:
                    camera_name = str(pose_entry['camera_name'])
                else:
                    image_path = pose_entry.get('image_path')
                    if image_path:
                        camera_name = Path(image_path).stem

                if camera_name is None:
                    logger.warning(
                        "Could not determine camera name for pose entry with keys: %s",
                        list(pose_entry.keys())
                    )
                    continue

                converted = convert_entry(camera_name, pose_entry)
                if converted:
                    if camera_name in normalized:
                        logger.warning("Duplicate pose entry for %s encountered, overwriting previous value", camera_name)
                    normalized[camera_name] = converted

            return normalized

        logger.warning("Unexpected camera_poses data type %s; returning empty dict", type(poses_data))
        return {}

    def _build_camera_id_lookup(self) -> Dict[int, str]:
        """Create mapping from numeric camera IDs to ground-truth camera names."""
        lookup: Dict[int, str] = {}

        for camera_name in self.ground_truth.cameras.keys():
            camera_index = self._parse_camera_index(camera_name)
            if camera_index is not None and camera_index not in lookup:
                lookup[camera_index] = camera_name

        if not lookup:
            logger.warning("Failed to build camera_id lookup from ground truth camera names")

        return lookup

    @staticmethod
    def _parse_camera_index(camera_name: str) -> Optional[int]:
        """Extract trailing integer from camera name like 'camera_012'."""
        match = re.search(r'(\d+)$', camera_name)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
        return None

    def _assess_overall_quality(self, results: Dict[str, Any]) -> bool:
        """Assess overall reconstruction quality."""
        success_criteria = {
            'pose_estimation': lambda r: r.get('pose_estimation', {}).get('overall', {}).get('mean_rotation_error', float('inf')) < 10.0,
            'triangulation': lambda r: r.get('triangulation', {}).get('overall', {}).get('mean_distance_error', float('inf')) < 0.2,
            'bundle_adjustment': lambda r: r.get('bundle_adjustment', {}).get('convergence', {}).get('converged', False)
        }

        passed_tests = []
        for test_name, test_func in success_criteria.items():
            passed = test_func(results)
            passed_tests.append(passed)
            logger.info(f"Quality check {test_name}: {'PASSED' if passed else 'FAILED'}")

        overall_success = sum(passed_tests) >= len(passed_tests) // 2
        logger.info(f"Overall quality assessment: {'SUCCESS' if overall_success else 'FAILED'}")

        return overall_success

    def save_evaluation_report(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save comprehensive evaluation report."""
        try:
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)

            logger.info(f"Evaluation report saved to {output_path}")

        except Exception as e:
            logger.error(f"Failed to save evaluation report: {e}")
            raise
