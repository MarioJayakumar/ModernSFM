"""
Camera trajectory generation with known poses for synthetic datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class CameraGenerator:
    """Generate camera trajectories with known ground truth poses."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.trajectory = config.get('trajectory', 'circular')
        self.num_cameras = config.get('num_cameras', 8)
        self.radius = config.get('radius', 5.0)
        self.height = config.get('height', 0.0)
        self.look_at = np.array(config.get('look_at', [0.0, 0.0, 0.0]))

        # Camera intrinsics
        intrinsics_config = config.get('intrinsics', {})
        self.focal_length = intrinsics_config.get('focal_length', 800.0)
        self.image_size = intrinsics_config.get('image_size', [640, 480])
        self.principal_point = intrinsics_config.get('principal_point', None)

        # Set default principal point to image center
        if self.principal_point is None:
            self.principal_point = [self.image_size[0] / 2.0, self.image_size[1] / 2.0]

    def generate_cameras(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """
        Generate camera poses and intrinsics.

        Returns:
            rotations: List of 3x3 rotation matrices
            translations: List of 3x1 translation vectors
            intrinsics: Camera intrinsics dictionary
        """
        logger.info(f"Generating {self.trajectory} trajectory with {self.num_cameras} cameras")

        if self.trajectory == 'circular':
            return self._generate_circular_trajectory()
        elif self.trajectory == 'linear':
            return self._generate_linear_trajectory()
        elif self.trajectory == 'random':
            return self._generate_random_trajectory()
        elif self.trajectory == 'arc':
            return self._generate_arc_trajectory()
        else:
            raise ValueError(f"Unknown trajectory type: {self.trajectory}")

    def _generate_circular_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """Generate cameras in a circular trajectory around the scene."""
        rotations = []
        translations = []

        for i in range(self.num_cameras):
            # Angle around the circle
            angle = 2 * np.pi * i / self.num_cameras

            # Camera position
            x = self.radius * np.cos(angle)
            y = self.height
            z = self.radius * np.sin(angle)
            camera_pos = np.array([x, y, z])

            # Camera looks at the center
            R, t = self._look_at_matrix(camera_pos, self.look_at, np.array([0, 1, 0]))

            rotations.append(R)
            translations.append(t)

        intrinsics = self._get_intrinsics_dict()

        logger.info(f"Generated circular trajectory: radius={self.radius}, height={self.height}")
        return rotations, translations, intrinsics

    def _generate_linear_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """Generate cameras along a linear path."""
        rotations = []
        translations = []

        # Linear path from left to right
        start_pos = np.array([-self.radius, self.height, self.radius])
        end_pos = np.array([self.radius, self.height, self.radius])

        for i in range(self.num_cameras):
            # Interpolate along the line
            t = i / (self.num_cameras - 1) if self.num_cameras > 1 else 0
            camera_pos = start_pos + t * (end_pos - start_pos)

            # Camera looks at the center
            R, translation = self._look_at_matrix(camera_pos, self.look_at, np.array([0, 1, 0]))

            rotations.append(R)
            translations.append(translation)

        intrinsics = self._get_intrinsics_dict()

        logger.info(f"Generated linear trajectory from {start_pos} to {end_pos}")
        return rotations, translations, intrinsics

    def _generate_arc_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """Generate cameras along an arc (partial circle)."""
        rotations = []
        translations = []

        # Arc from -90 to +90 degrees (180 degree arc)
        start_angle = -np.pi / 2
        end_angle = np.pi / 2

        for i in range(self.num_cameras):
            # Interpolate angle along the arc
            t = i / (self.num_cameras - 1) if self.num_cameras > 1 else 0
            angle = start_angle + t * (end_angle - start_angle)

            # Camera position
            x = self.radius * np.cos(angle)
            y = self.height
            z = self.radius * np.sin(angle)
            camera_pos = np.array([x, y, z])

            # Camera looks at the center
            R, translation = self._look_at_matrix(camera_pos, self.look_at, np.array([0, 1, 0]))

            rotations.append(R)
            translations.append(translation)

        intrinsics = self._get_intrinsics_dict()

        logger.info(f"Generated arc trajectory: {np.degrees(start_angle):.1f}° to {np.degrees(end_angle):.1f}°")
        return rotations, translations, intrinsics

    def _generate_random_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], Dict[str, float]]:
        """Generate cameras at random positions around the scene."""
        rotations = []
        translations = []

        # Set random seed for reproducible results
        np.random.seed(42)

        for i in range(self.num_cameras):
            # Random position on a sphere around the look_at point
            phi = np.random.uniform(0, 2 * np.pi)  # Azimuth
            theta = np.random.uniform(np.pi/6, 5*np.pi/6)  # Elevation (avoid poles)

            # Spherical to Cartesian
            x = self.radius * np.sin(theta) * np.cos(phi)
            y = self.radius * np.cos(theta) + self.height
            z = self.radius * np.sin(theta) * np.sin(phi)
            camera_pos = np.array([x, y, z])

            # Camera looks at the center with random up vector variation
            up_variation = np.random.uniform(-0.2, 0.2, 3)
            up_vector = np.array([0, 1, 0]) + up_variation
            up_vector = up_vector / np.linalg.norm(up_vector)

            R, t = self._look_at_matrix(camera_pos, self.look_at, up_vector)

            rotations.append(R)
            translations.append(t)

        intrinsics = self._get_intrinsics_dict()

        logger.info(f"Generated random trajectory: {self.num_cameras} cameras around radius {self.radius}")
        return rotations, translations, intrinsics

    def _look_at_matrix(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a look-at camera matrix.

        Args:
            eye: Camera position
            target: Point to look at
            up: Up vector

        Returns:
            R: 3x3 rotation matrix
            t: 3x1 translation vector
        """
        # Calculate camera coordinate system
        forward = target - eye
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, up)
        if np.linalg.norm(right) < 1e-6:  # Handle degenerate case
            right = np.array([1, 0, 0])
        else:
            right = right / np.linalg.norm(right)

        up_corrected = np.cross(right, forward)
        up_corrected = up_corrected / np.linalg.norm(up_corrected)

        # Ensure right-handed coordinate system by checking determinant
        test_R = np.array([right, up_corrected, forward])
        if np.linalg.det(test_R) < 0:
            # Flip one axis to make it right-handed
            up_corrected = -up_corrected

        # Create rotation matrix (world to camera)
        # In camera coordinates: X=right, Y=up, Z=forward (into scene)
        R = np.array([
            right,
            up_corrected,
            forward  # Camera looks down +z (forward into scene)
        ])

        # Translation vector
        t = -R @ eye

        return R, t

    def _get_intrinsics_dict(self) -> Dict[str, float]:
        """Get camera intrinsics as dictionary."""
        return {
            'fx': self.focal_length,
            'fy': self.focal_length,  # Assuming square pixels
            'cx': self.principal_point[0],
            'cy': self.principal_point[1]
        }

    def get_camera_names(self) -> List[str]:
        """Generate camera names for the trajectory."""
        return [f"camera_{i:03d}" for i in range(self.num_cameras)]

    def validate_trajectory(self, rotations: List[np.ndarray], translations: List[np.ndarray]) -> bool:
        """Validate that the generated trajectory has good geometry."""
        try:
            # Check that we have the right number of cameras
            assert len(rotations) == self.num_cameras
            assert len(translations) == self.num_cameras

            # Check rotation matrices are valid
            for i, R in enumerate(rotations):
                assert R.shape == (3, 3)
                # Check orthogonality
                assert np.allclose(R @ R.T, np.eye(3), atol=1e-6), f"Camera {i}: R is not orthogonal"
                # Check determinant is 1 (proper rotation)
                assert np.allclose(np.linalg.det(R), 1.0, atol=1e-6), f"Camera {i}: det(R) != 1"

            # Check translations have reasonable baselines
            positions = []
            for R, t in zip(rotations, translations):
                # Convert from camera coordinate to world coordinate
                position = -R.T @ t
                positions.append(position)

            positions = np.array(positions)

            # Check minimum baseline
            min_distance = float('inf')
            for i in range(len(positions)):
                for j in range(i+1, len(positions)):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_distance = min(min_distance, dist)

            assert min_distance > 0.1, f"Minimum baseline too small: {min_distance}"

            logger.info(f"Trajectory validation passed: min baseline = {min_distance:.3f}")
            return True

        except Exception as e:
            logger.error(f"Trajectory validation failed: {e}")
            return False