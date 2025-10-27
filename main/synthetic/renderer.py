"""
Synthetic image rendering pipeline for ground truth dataset generation.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SyntheticRenderer:
    """Render synthetic images from 3D scenes with known camera poses."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.lighting = config.get('lighting', 'uniform')
        self.background = config.get('background', 'black')
        self.noise_level = config.get('noise_level', 0.0)
        self.anti_aliasing = config.get('anti_aliasing', True)

    def render_images(self,
                     points_3d: np.ndarray,
                     rotations: List[np.ndarray],
                     translations: List[np.ndarray],
                     intrinsics: Dict[str, float],
                     camera_names: List[str],
                     output_dir: Path) -> Tuple[List[str], Dict[str, Dict[int, Tuple[float, float]]]]:
        """
        Render synthetic images from multiple camera viewpoints.

        Args:
            points_3d: Nx3 array of 3D points
            rotations: List of camera rotation matrices
            translations: List of camera translation vectors
            intrinsics: Camera intrinsics dict
            camera_names: List of camera names
            output_dir: Output directory for images

        Returns:
            image_paths: List of rendered image paths
            correspondences: Dict mapping camera_name -> {point_id: (px, py)}
        """
        logger.info(f"Rendering {len(camera_names)} images to {output_dir}")

        # Create output directory
        images_dir = output_dir / 'images'
        images_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        correspondences = {}

        # Get image size from intrinsics
        image_width = int(2 * intrinsics['cx'])
        image_height = int(2 * intrinsics['cy'])

        for i, (R, t, cam_name) in enumerate(zip(rotations, translations, camera_names)):
            logger.info(f"Rendering camera {i+1}/{len(camera_names)}: {cam_name}")

            # Render image
            image, visible_points, pixel_coords = self._render_single_image(
                points_3d, R, t, intrinsics, image_width, image_height
            )

            # Save image
            image_path = images_dir / f"{cam_name}.png"
            cv2.imwrite(str(image_path), image)
            image_paths.append(f"images/{cam_name}.png")

            # Store correspondences
            cam_correspondences = {}
            for point_id, (px, py) in zip(visible_points, pixel_coords):
                cam_correspondences[int(point_id)] = (float(px), float(py))

            correspondences[cam_name] = cam_correspondences

            logger.info(f"  Rendered {len(visible_points)} visible points")

        logger.info(f"Successfully rendered {len(image_paths)} images")
        return image_paths, correspondences

    def _render_single_image(self,
                           points_3d: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray,
                           intrinsics: Dict[str, float],
                           width: int,
                           height: int) -> Tuple[np.ndarray, List[int], List[Tuple[float, float]]]:
        """Render a single image from one camera viewpoint."""

        # Project 3D points to 2D
        visible_points, pixel_coords, depths = self._project_points(
            points_3d, R, t, intrinsics, width, height
        )

        # Create image
        image = self._create_base_image(width, height)

        # Add rendered points
        image = self._draw_points(image, pixel_coords, depths, visible_points, points_3d)

        # Apply post-processing
        if self.anti_aliasing:
            image = cv2.bilateralFilter(image, 9, 75, 75)

        if self.noise_level > 0:
            image = self._add_noise(image, self.noise_level)

        return image, visible_points, pixel_coords

    def _project_points(self,
                       points_3d: np.ndarray,
                       R: np.ndarray,
                       t: np.ndarray,
                       intrinsics: Dict[str, float],
                       width: int,
                       height: int) -> Tuple[List[int], List[Tuple[float, float]], List[float]]:
        """Project 3D points to 2D image coordinates."""

        # Transform points to camera coordinates
        points_cam = (R @ points_3d.T).T + t.T

        # Filter points behind camera
        valid_depth_mask = points_cam[:, 2] > 0.1  # Must be in front of camera

        # Project to image plane
        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']

        pixel_x = fx * points_cam[:, 0] / points_cam[:, 2] + cx
        pixel_y = fy * points_cam[:, 1] / points_cam[:, 2] + cy

        # Filter points within image bounds
        valid_x_mask = (pixel_x >= 0) & (pixel_x < width)
        valid_y_mask = (pixel_y >= 0) & (pixel_y < height)
        valid_mask = valid_depth_mask & valid_x_mask & valid_y_mask

        # Get visible points
        visible_indices = np.where(valid_mask)[0]
        visible_points = visible_indices.tolist()
        pixel_coords = [(pixel_x[i], pixel_y[i]) for i in visible_indices]
        depths = [points_cam[i, 2] for i in visible_indices]

        return visible_points, pixel_coords, depths

    def _create_base_image(self, width: int, height: int) -> np.ndarray:
        """Create base image with background."""
        if self.background == 'black':
            image = np.zeros((height, width, 3), dtype=np.uint8)
        elif self.background == 'white':
            image = np.full((height, width, 3), 255, dtype=np.uint8)
        elif self.background == 'gray':
            image = np.full((height, width, 3), 128, dtype=np.uint8)
        elif self.background == 'checkerboard':
            image = self._create_checkerboard_background(width, height)
        else:
            # Default to black
            image = np.zeros((height, width, 3), dtype=np.uint8)

        return image

    def _create_checkerboard_background(self, width: int, height: int) -> np.ndarray:
        """Create a checkerboard pattern background."""
        square_size = 50
        image = np.zeros((height, width, 3), dtype=np.uint8)

        for y in range(0, height, square_size):
            for x in range(0, width, square_size):
                if ((x // square_size) + (y // square_size)) % 2 == 0:
                    color = 200  # Light gray
                else:
                    color = 50   # Dark gray

                y_end = min(y + square_size, height)
                x_end = min(x + square_size, width)
                image[y:y_end, x:x_end] = color

        return image

    def _draw_points(self,
                    image: np.ndarray,
                    pixel_coords: List[Tuple[float, float]],
                    depths: List[float],
                    visible_points: List[int],
                    points_3d: np.ndarray) -> np.ndarray:
        """Draw 3D points on the image."""

        # Sort by depth (far to near) for proper occlusion
        depth_order = np.argsort(depths)[::-1]

        for idx in depth_order:
            px, py = pixel_coords[idx]
            point_id = visible_points[idx]

            # Determine point appearance based on type
            if point_id < 8:  # Cube vertices
                color = (0, 255, 0)  # Green for vertices
                radius = 8
            else:  # Texture points
                color = self._get_texture_color(point_id, points_3d[point_id])
                radius = 4

            # Draw point
            center = (int(px), int(py))
            cv2.circle(image, center, radius, color, -1)

            # Add subtle highlight for better visibility
            cv2.circle(image, center, radius + 1, (255, 255, 255), 1)

        return image

    def _get_texture_color(self, point_id: int, point_3d: np.ndarray) -> Tuple[int, int, int]:
        """Get color for texture points based on position."""
        # Derive a deterministic intensity from point coordinates to create unique local appearance
        x, y, z = point_3d
        # Hash coordinates into [0, 1)
        hash_val = (np.sin(x * 12.9898) + np.cos(y * 78.233) + np.sin(z * 37.719)) * 43758.5453
        hash_val = hash_val - np.floor(hash_val)
        intensity = 60 + int(hash_val * 180)  # Keep within visible grayscale range
        intensity = max(0, min(255, intensity))
        return (intensity, intensity, intensity)  # Grayscale color for SuperPoint compatibility

    def _add_noise(self, image: np.ndarray, noise_level: float) -> np.ndarray:
        """Add Gaussian noise to the image."""
        noise_std = noise_level * 255
        noise = np.random.normal(0, noise_std, image.shape).astype(np.float32)

        # Add noise and clamp to valid range
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)

        return noisy_image

    def visualize_projection(self,
                           points_3d: np.ndarray,
                           R: np.ndarray,
                           t: np.ndarray,
                           intrinsics: Dict[str, float],
                           output_path: Path) -> None:
        """Create a visualization showing the 3D scene and camera."""
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            fig = plt.figure(figsize=(12, 5))

            # 3D scene visualization
            ax1 = fig.add_subplot(121, projection='3d')
            ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                       c='blue', s=50, alpha=0.6)

            # Draw camera
            camera_pos = -R.T @ t
            ax1.scatter(*camera_pos, c='red', s=100, marker='^')

            # Draw camera orientation
            axes_length = 1.0
            for i, color in enumerate(['red', 'green', 'blue']):
                direction = R.T[:, i] * axes_length
                ax1.plot([camera_pos[0], camera_pos[0] + direction[0]],
                        [camera_pos[1], camera_pos[1] + direction[1]],
                        [camera_pos[2], camera_pos[2] + direction[2]],
                        color=color, linewidth=2)

            ax1.set_xlabel('X')
            ax1.set_ylabel('Y')
            ax1.set_zlabel('Z')
            ax1.set_title('3D Scene with Camera')

            # 2D projection visualization
            ax2 = fig.add_subplot(122)
            width = int(2 * intrinsics['cx'])
            height = int(2 * intrinsics['cy'])

            visible_points, pixel_coords, depths = self._project_points(
                points_3d, R, t, intrinsics, width, height
            )

            if pixel_coords:
                px_coords = [coord[0] for coord in pixel_coords]
                py_coords = [coord[1] for coord in pixel_coords]
                ax2.scatter(px_coords, py_coords, c='blue', s=30, alpha=0.6)

            ax2.set_xlim(0, width)
            ax2.set_ylim(height, 0)  # Flip Y axis for image coordinates
            ax2.set_xlabel('X (pixels)')
            ax2.set_ylabel('Y (pixels)')
            ax2.set_title(f'2D Projection ({len(visible_points)} points)')
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            logger.info(f"Projection visualization saved to {output_path}")

        except ImportError:
            logger.warning("Matplotlib not available, skipping projection visualization")
        except Exception as e:
            logger.warning(f"Failed to create projection visualization: {e}")
