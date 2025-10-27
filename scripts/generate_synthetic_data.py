#!/usr/bin/env python3
"""
Generate synthetic datasets with ground truth for SfM pipeline evaluation.

Usage:
    python -m scripts.generate_synthetic_data --config config/synthetic/simple_cube.yaml
    python -m scripts.generate_synthetic_data --config config/synthetic/textured_plane.yaml
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import yaml
import shutil
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main.synthetic import (
    SceneGenerator,
    CameraGenerator,
    SyntheticRenderer,
    GroundTruthData,
    SceneGroundTruth,
    CameraGroundTruth,
    save_ground_truth
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        raise


def create_output_directory(config: dict) -> Path:
    """Create timestamped output directory for the dataset."""
    base_path = Path(config['output']['base_path'])
    dataset_name = config['output']['dataset_name']
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_dir = base_path / f"{dataset_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created output directory: {output_dir}")
    return output_dir


def generate_synthetic_dataset(config: dict, output_dir: Path, config_path: Path) -> None:
    """Generate complete synthetic dataset with ground truth."""

    logger.info("Starting synthetic dataset generation")

    try:
        # Generate 3D scene
        logger.info("Generating 3D scene...")
        scene_generator = SceneGenerator(config['scene'])
        points_3d, point_ids, scene_params = scene_generator.generate_scene()
        scene_bounds = scene_generator.get_scene_bounds(points_3d)

        # Generate camera trajectory
        logger.info("Generating camera trajectory...")
        camera_generator = CameraGenerator(config['cameras'])
        rotations, translations, intrinsics = camera_generator.generate_cameras()
        camera_names = camera_generator.get_camera_names()

        # Validate camera trajectory
        if not camera_generator.validate_trajectory(rotations, translations):
            raise ValueError("Generated camera trajectory failed validation")

        # Render synthetic images
        logger.info("Rendering synthetic images...")
        renderer = SyntheticRenderer(config['rendering'])
        image_paths, correspondences = renderer.render_images(
            points_3d, rotations, translations, intrinsics, camera_names, output_dir
        )

        # Create ground truth data structure
        logger.info("Creating ground truth data...")

        # Scene ground truth
        scene_gt = SceneGroundTruth(
            points_3d=points_3d,
            point_ids=point_ids,
            scene_bounds=scene_bounds,
            scene_type=config['scene']['type'],
            scene_params=scene_params
        )

        # Camera ground truth
        cameras_gt = {}
        for i, (cam_name, image_path) in enumerate(zip(camera_names, image_paths)):
            visible_points = list(correspondences[cam_name].keys())

            cameras_gt[cam_name] = CameraGroundTruth(
                image_path=image_path,
                intrinsics=intrinsics,
                rotation=rotations[i],
                translation=translations[i],
                visible_points=visible_points,
                correspondences=correspondences[cam_name]
            )

        # Complete ground truth
        ground_truth = GroundTruthData(
            scene=scene_gt,
            cameras=cameras_gt,
            generation_config=config,
            timestamp=datetime.now().isoformat()
        )

        # Save ground truth data
        ground_truth_path = output_dir / 'ground_truth.json'
        save_ground_truth(ground_truth, ground_truth_path)

        # Copy configuration file for reference
        config_copy_path = output_dir / 'scene_config.yaml'
        shutil.copy2(config_path, config_copy_path)

        # Create projection visualization for debugging
        if len(rotations) > 0:
            viz_path = output_dir / 'projection_visualization.png'
            renderer.visualize_projection(
                points_3d, rotations[0], translations[0], intrinsics, viz_path
            )

        # Print summary
        logger.info("Synthetic dataset generation completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Generated {len(image_paths)} images")
        logger.info(f"Scene contains {len(points_3d)} 3D points")
        logger.info(f"Camera trajectory: {config['cameras']['trajectory']}")

        print("\n" + "="*60)
        print("SYNTHETIC DATASET GENERATION COMPLETE")
        print("="*60)
        print(f"Dataset: {config['output']['dataset_name']}")
        print(f"Output: {output_dir}")
        print(f"Images: {len(image_paths)}")
        print(f"3D Points: {len(points_3d)}")
        print(f"Scene Type: {config['scene']['type']}")
        print(f"Trajectory: {config['cameras']['trajectory']}")
        print("\nTo run reconstruction:")
        print(f"  python -m scripts.run_reconstruction --input {output_dir}/images/")
        print("\nTo evaluate results:")
        print(f"  python -m scripts.test_synthetic_reconstruction \\")
        print(f"    --ground_truth {output_dir}/ground_truth.json \\")
        print(f"    --reconstruction outputs/reconstruction_TIMESTAMP/")
        print("="*60)

    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        raise


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets with ground truth for SfM evaluation"
    )
    parser.add_argument(
        '--config',
        type=Path,
        required=True,
        help='Path to synthetic scene configuration file'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        help='Override output directory (optional)'
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = create_output_directory(config)

    # Generate dataset
    generate_synthetic_dataset(config, output_dir, args.config)


if __name__ == "__main__":
    main()