"""
3D scene generation with known geometry for synthetic datasets.
"""

import numpy as np
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SceneGenerator:
    """Generate synthetic 3D scenes with known ground truth geometry."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.scene_type = config.get('type', 'cube')
        self.size = config.get('size', 2.0)
        self.position = np.array(config.get('position', [0.0, 0.0, 0.0]))
        self.texture_type = config.get('texture_type', 'checkerboard')
        self.texture_scale = config.get('texture_scale', 0.1)
        self.texture_jitter = config.get('texture_jitter', 0.05)
        self.interior_points = config.get('interior_points', 64)

    def generate_scene(self) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """
        Generate 3D scene geometry.

        Returns:
            points_3d: Nx3 array of world coordinates
            point_ids: List of unique point IDs
            scene_params: Dictionary of scene parameters
        """
        logger.info(f"Generating {self.scene_type} scene with size {self.size}")

        if self.scene_type == 'cube':
            return self._generate_cube()
        elif self.scene_type == 'textured_plane':
            return self._generate_textured_plane()
        elif self.scene_type == 'multi_object':
            return self._generate_multi_object()
        else:
            raise ValueError(f"Unknown scene type: {self.scene_type}")

    def _generate_cube(self) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """Generate a textured cube with known vertex positions."""
        half_size = self.size / 2.0

        # Define cube vertices
        vertices = np.array([
            # Front face
            [-half_size, -half_size,  half_size],  # 0: bottom-left
            [ half_size, -half_size,  half_size],  # 1: bottom-right
            [ half_size,  half_size,  half_size],  # 2: top-right
            [-half_size,  half_size,  half_size],  # 3: top-left
            # Back face
            [-half_size, -half_size, -half_size],  # 4: bottom-left
            [ half_size, -half_size, -half_size],  # 5: bottom-right
            [ half_size,  half_size, -half_size],  # 6: top-right
            [-half_size,  half_size, -half_size],  # 7: top-left
        ])

        # Add texture points on faces for better feature detection
        texture_points = []
        texture_id_offset = 8

        if self.texture_type == 'checkerboard':
            # Add checkerboard pattern points on each face
            n_points_per_side = int(1.0 / self.texture_scale)
            jitter_amount = self.texture_jitter * self.size
            jitter_amount = min(jitter_amount, self.size / (4 * n_points_per_side + 1))

            # Front face (z = half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = -half_size + (i + 0.5) * self.size / n_points_per_side
                    y = -half_size + (j + 0.5) * self.size / n_points_per_side
                    x += self._face_jitter(0, i, j, jitter_amount)
                    y += self._face_jitter(1, i, j, jitter_amount)
                    z = half_size
                    texture_points.append([x, y, z])

            # Back face (z = -half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = -half_size + (i + 0.5) * self.size / n_points_per_side
                    y = -half_size + (j + 0.5) * self.size / n_points_per_side
                    x += self._face_jitter(2, i, j, jitter_amount)
                    y += self._face_jitter(3, i, j, jitter_amount)
                    z = -half_size
                    texture_points.append([x, y, z])

            # Left face (x = -half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = -half_size
                    y = -half_size + (i + 0.5) * self.size / n_points_per_side
                    z = -half_size + (j + 0.5) * self.size / n_points_per_side
                    y += self._face_jitter(4, i, j, jitter_amount)
                    z += self._face_jitter(5, i, j, jitter_amount)
                    texture_points.append([x, y, z])

            # Right face (x = half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = half_size
                    y = -half_size + (i + 0.5) * self.size / n_points_per_side
                    z = -half_size + (j + 0.5) * self.size / n_points_per_side
                    y += self._face_jitter(6, i, j, jitter_amount)
                    z += self._face_jitter(7, i, j, jitter_amount)
                    texture_points.append([x, y, z])

            # Top face (y = half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = -half_size + (i + 0.5) * self.size / n_points_per_side
                    y = half_size
                    z = -half_size + (j + 0.5) * self.size / n_points_per_side
                    x += self._face_jitter(8, i, j, jitter_amount)
                    z += self._face_jitter(9, i, j, jitter_amount)
                    texture_points.append([x, y, z])

            # Bottom face (y = -half_size)
            for i in range(n_points_per_side):
                for j in range(n_points_per_side):
                    x = -half_size + (i + 0.5) * self.size / n_points_per_side
                    y = -half_size
                    z = -half_size + (j + 0.5) * self.size / n_points_per_side
                    x += self._face_jitter(10, i, j, jitter_amount)
                    z += self._face_jitter(11, i, j, jitter_amount)
                    texture_points.append([x, y, z])

        # Optionally sprinkle interior feature points to break planar degeneracies
        if self.interior_points > 0:
            rng = np.random.default_rng(42)
            for idx in range(self.interior_points):
                interior = rng.uniform(-half_size, half_size, size=3)
                texture_points.append(interior.tolist())

        # Combine vertices and texture points
        all_points = np.vstack([vertices, np.array(texture_points)])

        # Translate to desired position
        all_points += self.position

        # Generate point IDs
        point_ids = list(range(len(all_points)))

        scene_params = {
            'scene_type': self.scene_type,
            'size': self.size,
            'position': self.position.tolist(),
            'texture_type': self.texture_type,
            'texture_scale': self.texture_scale,
            'n_vertices': len(vertices),
            'n_texture_points': len(texture_points),
            'n_interior_points': self.interior_points,
            'n_total_points': len(all_points)
        }

        logger.info(f"Generated cube: {len(vertices)} vertices, {len(texture_points)} texture points")
        return all_points, point_ids, scene_params

    @staticmethod
    def _hash_noise(a: int, b: int, c: int) -> float:
        """Deterministic pseudo-random number in [0, 1)."""
        value = (a * 73856093) ^ (b * 19349663) ^ (c * 83492791)
        value = value & 0xFFFFFFFF
        return (value % 1000003) / 1000003.0

    def _face_jitter(self, face_id: int, i: int, j: int, jitter_amount: float) -> float:
        """Deterministic in-plane jitter to break symmetric feature layouts."""
        if jitter_amount <= 0:
            return 0.0
        noise = self._hash_noise(face_id, i, j) - 0.5
        return noise * 2.0 * jitter_amount

    def _generate_textured_plane(self) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """Generate a textured plane for testing planar scenes."""
        half_size = self.size / 2.0

        # Generate grid of points on XY plane (z=0)
        n_points_per_side = int(1.0 / self.texture_scale)
        points = []

        for i in range(n_points_per_side + 1):
            for j in range(n_points_per_side + 1):
                x = -half_size + i * self.size / n_points_per_side
                y = -half_size + j * self.size / n_points_per_side
                z = 0.0
                points.append([x, y, z])

        points = np.array(points) + self.position
        point_ids = list(range(len(points)))

        scene_params = {
            'scene_type': self.scene_type,
            'size': self.size,
            'position': self.position.tolist(),
            'texture_scale': self.texture_scale,
            'n_points': len(points)
        }

        logger.info(f"Generated textured plane: {len(points)} points")
        return points, point_ids, scene_params

    def _generate_multi_object(self) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """Generate multiple objects for complex scene testing."""
        all_points = []
        point_id_counter = 0

        # Generate cube
        cube_config = {
            'type': 'cube',
            'size': self.size * 0.7,
            'position': [-self.size * 0.5, 0, 0],
            'texture_type': self.texture_type,
            'texture_scale': self.texture_scale
        }
        cube_gen = SceneGenerator(cube_config)
        cube_points, _, _ = cube_gen._generate_cube()

        # Generate textured plane
        plane_config = {
            'type': 'textured_plane',
            'size': self.size * 0.8,
            'position': [self.size * 0.5, 0, 0],
            'texture_scale': self.texture_scale
        }
        plane_gen = SceneGenerator(plane_config)
        plane_points, _, _ = plane_gen._generate_textured_plane()

        # Combine all points
        all_points = np.vstack([cube_points, plane_points])
        point_ids = list(range(len(all_points)))

        scene_params = {
            'scene_type': self.scene_type,
            'objects': ['cube', 'textured_plane'],
            'n_cube_points': len(cube_points),
            'n_plane_points': len(plane_points),
            'n_total_points': len(all_points)
        }

        logger.info(f"Generated multi-object scene: {len(all_points)} total points")
        return all_points, point_ids, scene_params

    def get_scene_bounds(self, points_3d: np.ndarray) -> Dict[str, List[float]]:
        """Calculate bounding box of the scene."""
        min_bounds = points_3d.min(axis=0).tolist()
        max_bounds = points_3d.max(axis=0).tolist()

        return {
            'min': min_bounds,
            'max': max_bounds
        }
