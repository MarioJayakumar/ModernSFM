"""
Ground truth data structures and I/O for synthetic datasets.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class CameraGroundTruth:
    """Ground truth data for a single camera."""
    image_path: str
    intrinsics: Dict[str, float]  # fx, fy, cx, cy
    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # 3x1 translation vector
    visible_points: List[int]  # IDs of visible 3D points
    correspondences: Dict[int, Tuple[float, float]]  # point_id -> (px, py)


@dataclass
class SceneGroundTruth:
    """Ground truth data for the 3D scene."""
    points_3d: np.ndarray  # Nx3 world coordinates
    point_ids: List[int]  # Unique point IDs
    scene_bounds: Dict[str, List[float]]  # min/max bounds
    scene_type: str  # cube, sphere, etc.
    scene_params: Dict[str, Any]  # generation parameters


@dataclass
class GroundTruthData:
    """Complete ground truth dataset."""
    scene: SceneGroundTruth
    cameras: Dict[str, CameraGroundTruth]
    generation_config: Dict[str, Any]
    timestamp: str

    def get_camera_poses_matrix(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get all camera poses as matrices for easy processing."""
        n_cameras = len(self.cameras)
        rotations = np.zeros((n_cameras, 3, 3))
        translations = np.zeros((n_cameras, 3))

        for i, (cam_name, cam_data) in enumerate(sorted(self.cameras.items())):
            rotations[i] = cam_data.rotation
            translations[i] = cam_data.translation

        return rotations, translations

    def get_intrinsics_matrix(self, camera_name: str) -> np.ndarray:
        """Get intrinsics matrix for a specific camera."""
        intrinsics = self.cameras[camera_name].intrinsics
        K = np.array([
            [intrinsics['fx'], 0, intrinsics['cx']],
            [0, intrinsics['fy'], intrinsics['cy']],
            [0, 0, 1]
        ])
        return K

    def get_all_correspondences(self) -> Dict[str, Dict[int, Tuple[float, float]]]:
        """Get all 2D-3D correspondences for evaluation."""
        correspondences = {}
        for cam_name, cam_data in self.cameras.items():
            correspondences[cam_name] = cam_data.correspondences
        return correspondences

    def get_visibility_matrix(self) -> np.ndarray:
        """Get binary visibility matrix: cameras x points."""
        n_cameras = len(self.cameras)
        n_points = len(self.scene.point_ids)
        visibility = np.zeros((n_cameras, n_points), dtype=bool)

        for i, (cam_name, cam_data) in enumerate(sorted(self.cameras.items())):
            for point_id in cam_data.visible_points:
                if point_id in self.scene.point_ids:
                    j = self.scene.point_ids.index(point_id)
                    visibility[i, j] = True

        return visibility


def save_ground_truth(ground_truth: GroundTruthData, output_path: Path) -> None:
    """Save ground truth data to JSON file."""
    try:
        # Convert to serializable format
        data = {
            "scene": {
                "points_3d": ground_truth.scene.points_3d.tolist(),
                "point_ids": ground_truth.scene.point_ids,
                "scene_bounds": ground_truth.scene.scene_bounds,
                "scene_type": ground_truth.scene.scene_type,
                "scene_params": ground_truth.scene.scene_params
            },
            "cameras": {},
            "generation_config": ground_truth.generation_config,
            "timestamp": ground_truth.timestamp
        }

        # Convert camera data
        for cam_name, cam_data in ground_truth.cameras.items():
            data["cameras"][cam_name] = {
                "image_path": cam_data.image_path,
                "intrinsics": cam_data.intrinsics,
                "extrinsics": {
                    "rotation": cam_data.rotation.tolist(),
                    "translation": cam_data.translation.tolist()
                },
                "visible_points": cam_data.visible_points,
                "correspondences": {
                    str(point_id): [px, py]
                    for point_id, (px, py) in cam_data.correspondences.items()
                }
            }

        # Save to file
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Ground truth saved to {output_path}")

    except Exception as e:
        logger.error(f"Failed to save ground truth: {e}")
        raise


def load_ground_truth(input_path: Path) -> GroundTruthData:
    """Load ground truth data from JSON file."""
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)

        # Parse scene data
        scene = SceneGroundTruth(
            points_3d=np.array(data["scene"]["points_3d"]),
            point_ids=data["scene"]["point_ids"],
            scene_bounds=data["scene"]["scene_bounds"],
            scene_type=data["scene"]["scene_type"],
            scene_params=data["scene"]["scene_params"]
        )

        # Parse camera data
        cameras = {}
        for cam_name, cam_data in data["cameras"].items():
            correspondences = {
                int(point_id): tuple(coords)
                for point_id, coords in cam_data["correspondences"].items()
            }

            cameras[cam_name] = CameraGroundTruth(
                image_path=cam_data["image_path"],
                intrinsics=cam_data["intrinsics"],
                rotation=np.array(cam_data["extrinsics"]["rotation"]),
                translation=np.array(cam_data["extrinsics"]["translation"]),
                visible_points=cam_data["visible_points"],
                correspondences=correspondences
            )

        ground_truth = GroundTruthData(
            scene=scene,
            cameras=cameras,
            generation_config=data["generation_config"],
            timestamp=data["timestamp"]
        )

        logger.info(f"Ground truth loaded from {input_path}")
        logger.info(f"Scene: {len(scene.points_3d)} points, {len(cameras)} cameras")

        return ground_truth

    except Exception as e:
        logger.error(f"Failed to load ground truth: {e}")
        raise


def validate_ground_truth(ground_truth: GroundTruthData) -> bool:
    """Validate ground truth data consistency."""
    try:
        # Check scene data
        assert len(ground_truth.scene.points_3d) == len(ground_truth.scene.point_ids)
        assert ground_truth.scene.points_3d.shape[1] == 3

        # Check camera data
        for cam_name, cam_data in ground_truth.cameras.items():
            # Check rotation matrix
            R = cam_data.rotation
            assert R.shape == (3, 3)
            assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-6)

            # Check translation
            assert cam_data.translation.shape == (3,)

            # Check correspondences
            for point_id, (px, py) in cam_data.correspondences.items():
                assert point_id in ground_truth.scene.point_ids
                assert isinstance(px, (int, float)) and isinstance(py, (int, float))

        logger.info("Ground truth validation passed")
        return True

    except Exception as e:
        logger.error(f"Ground truth validation failed: {e}")
        return False