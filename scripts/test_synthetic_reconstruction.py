#!/usr/bin/env python3
"""
Test and evaluate SfM reconstruction against synthetic ground truth data.

Usage:
    python -m scripts.test_synthetic_reconstruction \
        --ground_truth data/synthetic/simple_cube_20241023_143022/ground_truth.json \
        --reconstruction outputs/reconstruction_20241023_143045/
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main.synthetic import SyntheticEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_inputs(ground_truth_path: Path, reconstruction_dir: Path) -> None:
    """Validate input paths exist and contain expected files."""

    # Check ground truth file
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")

    if not ground_truth_path.suffix == '.json':
        raise ValueError(f"Ground truth file must be JSON: {ground_truth_path}")

    # Check reconstruction directory
    if not reconstruction_dir.exists():
        raise FileNotFoundError(f"Reconstruction directory not found: {reconstruction_dir}")

    if not reconstruction_dir.is_dir():
        raise ValueError(f"Reconstruction path must be a directory: {reconstruction_dir}")

    # Look for expected reconstruction files
    expected_files = [
        'reconstruction_report.json',
        'data/points_3d.npy',
        'data/camera_poses.json'
    ]

    found_files = []
    missing_files = []
    search_roots = [reconstruction_dir]
    data_dir = reconstruction_dir / "data"
    if data_dir.exists():
        search_roots.append(data_dir)

    for filename in expected_files:
        filepath = None
        for root in search_roots:
            candidate = root / filename if root == reconstruction_dir else root / Path(filename).name
            if candidate.exists():
                filepath = candidate
                break

        expected_path = reconstruction_dir / filename
        if filepath:
            found_files.append(filepath)
        else:
            missing_files.append(str(expected_path.relative_to(reconstruction_dir)))

    if missing_files:
        logger.warning(
            "Missing expected reconstruction files: %s",
            missing_files
        )
        logger.debug(
            "Available files under reconstruction dir: %s",
            [str(p) for p in reconstruction_dir.rglob('*.*')]
        )

    if found_files:
        logger.info("Found reconstruction files:\n%s", "\n".join(f"  - {path}" for path in found_files))
    else:
        logger.warning("No expected reconstruction files found in %s", reconstruction_dir)


def load_ground_truth_summary(ground_truth_path: Path) -> dict:
    """Load and summarize ground truth data for display."""
    try:
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)

        summary = {
            'scene_type': gt_data.get('scene', {}).get('scene_type', 'unknown'),
            'n_points': len(gt_data.get('scene', {}).get('points_3d', [])),
            'n_cameras': len(gt_data.get('cameras', {})),
            'timestamp': gt_data.get('timestamp', 'unknown'),
            'config_file': gt_data.get('generation_config', {}).get('output', {}).get('dataset_name', 'unknown')
        }

        return summary

    except Exception as e:
        logger.error(f"Failed to load ground truth summary: {e}")
        return {}


def print_evaluation_summary(
    results: dict,
    ground_truth_path: Path,
    reconstruction_dir: Path
) -> None:
    """Print comprehensive evaluation summary."""
    print("\n" + "="*80)
    print("SYNTHETIC RECONSTRUCTION EVALUATION RESULTS")
    print("="*80)

    # Basic info
    print(f"Ground Truth Scene: {results.get('ground_truth_scene', 'unknown')}")
    print(f"Ground Truth File: {ground_truth_path}")
    print(f"Reconstruction Directory: {reconstruction_dir}")
    print(f"Evaluation Time: {results.get('timestamp', 'unknown')}")

    # Overall assessment
    summary = results.get('evaluation_summary', {})
    overall_success = summary.get('overall_success', False)
    print(f"\nOVERALL ASSESSMENT: {'✅ SUCCESS' if overall_success else '❌ FAILED'}")

    # Component results
    print(f"\n{'COMPONENT':<20} {'STATUS':<10} {'KEY METRIC'}")
    print("-" * 50)

    # Pose estimation
    if 'pose_estimation' in results:
        pose_results = results['pose_estimation']
        pose_passed = summary.get('pose_estimation_passed', False)
        mean_rot_error = pose_results.get('overall', {}).get('mean_rotation_error', float('inf'))
        status = '✅ PASS' if pose_passed else '❌ FAIL'
        print(f"{'Pose Estimation':<20} {status:<10} {mean_rot_error:.2f}° rotation error")

    # Triangulation
    if 'triangulation' in results:
        tri_results = results['triangulation']
        tri_passed = summary.get('triangulation_passed', False)
        mean_dist_error = tri_results.get('overall', {}).get('mean_distance_error', float('inf'))
        match_ratio = tri_results.get('overall', {}).get('match_ratio', 0.0)
        status = '✅ PASS' if tri_passed else '❌ FAIL'
        print(f"{'Triangulation':<20} {status:<10} {mean_dist_error:.4f} distance, {match_ratio:.2f} match ratio")

    # Bundle adjustment
    if 'bundle_adjustment' in results:
        ba_results = results['bundle_adjustment']
        ba_improved = summary.get('bundle_adjustment_improved', False)
        converged = ba_results.get('convergence', {}).get('converged', False)
        error_reduction = ba_results.get('convergence', {}).get('error_reduction', 0.0)
        status = '✅ PASS' if ba_improved else '❌ FAIL'
        print(f"{'Bundle Adjustment':<20} {status:<10} {'Converged' if converged else 'No convergence'}, {error_reduction:.4f} error reduction")

    print("\n" + "="*80)

    # Detailed metrics
    if overall_success:
        print("🎉 RECONSTRUCTION SUCCESSFUL!")
        print("The SfM pipeline successfully reconstructed the synthetic scene.")
    else:
        print("⚠️  RECONSTRUCTION ISSUES DETECTED")
        print("Review the detailed metrics below to identify bottlenecks.")

    print("\nDETAILED METRICS:")
    print("-" * 40)

    # Pose estimation details
    if 'pose_estimation' in results:
        pose = results['pose_estimation']['overall']
        print(f"\n📐 Pose Estimation:")
        print(f"  Mean rotation error: {pose.get('mean_rotation_error', 0):.2f}° ± {pose.get('std_rotation_error', 0):.2f}°")
        print(f"  Max rotation error:  {pose.get('max_rotation_error', 0):.2f}°")
        print(f"  Mean translation error: {pose.get('mean_translation_error', 0):.4f} ± {pose.get('std_translation_error', 0):.4f}")
        print(f"  Cameras evaluated: {pose.get('cameras_evaluated', 0)}")

    # Triangulation details
    if 'triangulation' in results:
        tri = results['triangulation']['overall']
        print(f"\n🔺 Triangulation:")
        print(f"  Mean distance error: {tri.get('mean_distance_error', 0):.4f} ± {tri.get('std_distance_error', 0):.4f}")
        print(f"  Match ratio: {tri.get('match_ratio', 0):.2f} ({tri.get('matched_points', 0)}/{results['triangulation']['n_gt_points']} points)")
        print(f"  Precision: {tri.get('precision', 0):.2f}")
        print(f"  Completeness: {tri.get('completeness', 0):.2f}")

    # Bundle adjustment details
    if 'bundle_adjustment' in results:
        ba = results['bundle_adjustment']['convergence']
        print(f"\n⚖️  Bundle Adjustment:")
        print(f"  Iterations: {ba.get('iterations', 0)}")
        print(f"  Initial error: {ba.get('initial_error', 0):.6f}")
        print(f"  Final error: {ba.get('final_error', 0):.6f}")
        print(f"  Error reduction: {ba.get('error_reduction', 0):.6f}")
        print(f"  Converged: {'Yes' if ba.get('converged', False) else 'No'}")

    print("\n" + "="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate SfM reconstruction results against synthetic ground truth"
    )
    parser.add_argument(
        '--ground_truth',
        type=Path,
        required=True,
        help='Path to ground truth JSON file'
    )
    parser.add_argument(
        '--reconstruction',
        type=Path,
        required=True,
        help='Path to reconstruction results directory'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for evaluation report (optional)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting synthetic reconstruction evaluation")

    try:
        # Validate inputs
        validate_inputs(args.ground_truth, args.reconstruction)

        resolved_ground_truth = args.ground_truth.resolve()
        resolved_reconstruction = args.reconstruction.resolve()

        logger.info("Using ground truth file: %s", resolved_ground_truth)
        logger.info("Using reconstruction directory: %s", resolved_reconstruction)

        # Load ground truth summary
        gt_summary = load_ground_truth_summary(resolved_ground_truth)
        logger.info(f"Ground truth: {gt_summary.get('scene_type', 'unknown')} scene with "
                   f"{gt_summary.get('n_points', 0)} points, {gt_summary.get('n_cameras', 0)} cameras")

        # Initialize evaluator
        evaluator = SyntheticEvaluator(resolved_ground_truth)

        # Run evaluation
        logger.info("Running comprehensive evaluation...")
        results = evaluator.evaluate_reconstruction(resolved_reconstruction)

        results.setdefault('inputs', {})
        results['inputs'].update({
            'ground_truth_path': str(resolved_ground_truth),
            'reconstruction_path': str(resolved_reconstruction)
        })

        # Print results
        print_evaluation_summary(results, resolved_ground_truth, resolved_reconstruction)

        # Save detailed report if requested
        if args.output:
            evaluator.save_evaluation_report(results, args.output)
            print(f"\n📄 Detailed report saved to: {args.output}")
        elif 'error' not in results:
            # Default output location
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(f"synthetic_evaluation_report_{timestamp}.json")
            evaluator.save_evaluation_report(results, output_path)
            print(f"\n📄 Detailed report saved to: {output_path}")

        # Exit with appropriate code
        if results.get('evaluation_summary', {}).get('overall_success', False):
            logger.info("Evaluation completed successfully - all tests passed!")
            sys.exit(0)
        else:
            logger.warning("Evaluation completed - some tests failed")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"\n❌ EVALUATION ERROR: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()
