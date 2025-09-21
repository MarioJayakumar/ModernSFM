"""
Command-line interface for running the full ModernSFM reconstruction pipeline.

This script provides a user-friendly interface to the complete SfM pipeline,
supporting various input/output options, quality settings, and visualization modes.

Usage:
    python -m scripts.run_reconstruction --input data/test_data/ --output outputs/my_reconstruction
    python -m scripts.run_reconstruction --input data/acropolis/ --name acropolis_recon --config config/base_config.yaml
"""

import argparse
import sys
from pathlib import Path
import logging
import os
import json
from datetime import datetime
from omegaconf import OmegaConf

# Add the main directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from main.pipeline.full_reconstruction import FullReconstructionPipeline, ReconstructionResult

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_output_directory(base_dir: str, name: str = None) -> Path:
    """
    Create timestamped output directory following project conventions.
    
    Args:
        base_dir: Base output directory (e.g., 'outputs')
        name: Optional name for the reconstruction
        
    Returns:
        Path to created output directory
    """
    base_path = Path(base_dir)
    
    # Create date subdirectory
    date_str = datetime.now().strftime("%Y-%m-%d")
    date_dir = base_path / date_str
    
    # Create time subdirectory  
    time_str = datetime.now().strftime("%H-%M-%S")
    if name:
        time_dir = date_dir / f"{time_str}_{name}"
    else:
        time_dir = date_dir / time_str
    
    time_dir.mkdir(parents=True, exist_ok=True)
    return time_dir


def setup_logging(output_dir: Path, name: str = "reconstruction") -> Path:
    """
    Set up file logging in addition to console logging.
    
    Args:
        output_dir: Directory for log file
        name: Base name for log file
        
    Returns:
        Path to log file
    """
    log_file = output_dir / f"{name}.log"
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    
    logger.info(f"Logging to file: {log_file}")
    return log_file


def print_summary(result: ReconstructionResult):
    """Print a comprehensive summary of reconstruction results."""
    print("\n" + "="*80)
    print("RECONSTRUCTION SUMMARY")
    print("="*80)
    
    # Pipeline status
    if result.success:
        print("✅ Status: SUCCESS")
    else:
        print("❌ Status: FAILED")
        if result.error_message:
            print(f"   Error: {result.error_message}")
        return
    
    # Input/output info
    print(f"\n📂 Input Images: {result.num_images}")
    if result.output_directory:
        print(f"📁 Output Directory: {result.output_directory}")
    
    # Pipeline stages
    print(f"\n🔧 Pipeline Stages:")
    total_features = sum(result.num_features_extracted.values())
    total_matches = sum(result.num_matches.values())
    
    print(f"   • Feature Extraction: {total_features} features across {result.num_images} images")
    print(f"   • Feature Matching: {total_matches} matches across {len(result.num_matches)} pairs")
    print(f"   • Pose Estimation: {result.num_poses_estimated}/{result.num_images} poses estimated")
    print(f"   • Triangulation: {result.num_points_triangulated} 3D points")
    print(f"   • Bundle Adjustment: {'Success' if result.bundle_adjustment_result else 'Failed'}")
    
    # Quality metrics
    print(f"\n📊 Quality Metrics:")
    if result.mean_reprojection_error is not None:
        print(f"   • Mean Reprojection Error: {result.mean_reprojection_error:.3f} pixels")
    if result.inlier_ratio is not None:
        print(f"   • Inlier Ratio: {result.inlier_ratio:.3f}")
    if result.reconstruction_scale is not None:
        print(f"   • Reconstruction Scale: {result.reconstruction_scale:.3f}")
    
    # Bundle adjustment details
    if result.bundle_adjustment_result:
        ba = result.bundle_adjustment_result
        print(f"   • BA RMSE Improvement: {ba.initial_rmse:.3f} → {ba.final_rmse:.3f}")
        print(f"   • BA Iterations: {ba.num_iterations}")
        print(f"   • BA Convergence: {ba.convergence_reason}")
    
    # Performance metrics
    print(f"\n⏱️ Performance:")
    print(f"   • Total Time: {result.total_time:.2f}s")
    if result.stage_times:
        for stage, time_taken in result.stage_times.items():
            print(f"   • {stage.replace('_', ' ').title()}: {time_taken:.2f}s")
    
    # Output files
    if result.visualization_files:
        print(f"\n🎨 Generated Visualizations:")
        for vis_file in result.visualization_files[:5]:  # Show first 5
            print(f"   • {vis_file}")
        if len(result.visualization_files) > 5:
            print(f"   • ... and {len(result.visualization_files) - 5} more files")
    
    print("="*80)


def save_reconstruction_report(result: ReconstructionResult, output_file: Path):
    """Save detailed reconstruction report as JSON."""
    try:
        # Convert result to serializable format
        report = {
            'success': result.success,
            'error_message': result.error_message,
            'timestamp': datetime.now().isoformat(),
            'pipeline_info': {
                'num_images': result.num_images,
                'num_features_extracted': result.num_features_extracted,
                'num_matches': {f"{k[0]}-{k[1]}": v for k, v in result.num_matches.items()},  # Convert tuple keys to strings
                'num_poses_estimated': result.num_poses_estimated,
                'num_points_triangulated': result.num_points_triangulated
            },
            'quality_metrics': {
                'mean_reprojection_error': result.mean_reprojection_error,
                'inlier_ratio': result.inlier_ratio,
                'reconstruction_scale': result.reconstruction_scale
            },
            'performance_metrics': {
                'total_time': result.total_time,
                'stage_times': result.stage_times
            },
            'output_info': {
                'output_directory': str(result.output_directory) if result.output_directory else None,
                'visualization_files': [str(f) for f in result.visualization_files] if result.visualization_files else []
            }
        }
        
        # Add bundle adjustment details if available
        if result.bundle_adjustment_result:
            ba = result.bundle_adjustment_result
            report['bundle_adjustment'] = {
                'initial_rmse': ba.initial_rmse,
                'final_rmse': ba.final_rmse,
                'num_iterations': ba.num_iterations,
                'convergence_reason': ba.convergence_reason,
                'inlier_ratio': ba.inlier_ratio,
                'optimization_time': ba.optimization_time
            }
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved reconstruction report to: {output_file}")
        
    except Exception as e:
        logger.error(f"Failed to save reconstruction report: {e}")


def main():
    """Main function using direct command line arguments."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Run complete ModernSFM reconstruction pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m scripts.run_reconstruction --input data/test_data/ --output outputs/test_recon
  python -m scripts.run_reconstruction --input data/acropolis/ --name acropolis --quality high
  python -m scripts.run_reconstruction --input /path/to/images --output /path/to/output --config custom_config.yaml
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Input directory containing images')
    parser.add_argument('--output', '-o', type=str, default='outputs',
                       help='Base output directory (default: outputs)')
    parser.add_argument('--name', '-n', type=str, default=None,
                       help='Name for this reconstruction (used in output directory)')
    parser.add_argument('--quality', '-q', choices=['fast', 'balanced', 'high'], default='balanced',
                       help='Quality preset: fast, balanced, or high (default: balanced)')
    parser.add_argument('--device', '-d', choices=['auto', 'cpu', 'cuda'], default='auto',
                       help='Device to use: auto, cpu, or cuda (default: auto)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--no-visualization', '--no-viz', action='store_true',
                       help='Skip all visualization generation')
    parser.add_argument('--intermediate-viz', action='store_true',
                       help='Generate intermediate visualizations (features, matches, triangulation)')
    parser.add_argument('--max-images', type=int, default=None,
                       help='Limit number of input images (for testing)')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    # Validate input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        sys.exit(1)
    
    # Create output directory
    output_dir = create_output_directory(args.output, args.name)
    logger.info(f"Output directory: {output_dir}")
    
    # Set up file logging
    reconstruction_name = args.name or "reconstruction"
    log_file = setup_logging(output_dir, reconstruction_name)
    
    # Load base configuration
    try:
        cfg = OmegaConf.load('config/base_config.yaml')
    except Exception as e:
        logger.error(f"Failed to load base configuration: {e}")
        sys.exit(1)
    
    # Apply quality preset to configuration
    if args.quality == 'fast':
        # Fast preset - reduce quality for speed
        cfg.feature_extractor.max_keypoints = 2000
        cfg.feature_matching.max_matches = 1000
        cfg.bundle_adjustment.max_iterations = 50
    elif args.quality == 'high':
        # High quality preset
        cfg.feature_extractor.max_keypoints = 12000
        cfg.feature_matching.max_matches = 5000
        cfg.bundle_adjustment.max_iterations = 500
    # 'balanced' uses default config values
    
    # Override device if specified
    if args.device != 'auto':
        cfg.device = args.device
    
    logger.info(f"Using quality preset: {args.quality}")
    logger.info(f"Using device: {cfg.device}")
    
    try:
        # Initialize and run pipeline
        logger.info("Initializing reconstruction pipeline...")
        pipeline = FullReconstructionPipeline(cfg)
        
        # Run reconstruction
        logger.info(f"Starting reconstruction of images in: {input_dir}")
        result = pipeline.reconstruct(
            image_dir=input_dir,
            output_dir=output_dir,
            name=reconstruction_name,
            max_images=args.max_images
        )
        
        # Print summary
        print_summary(result)
        
        # Save detailed report
        report_file = output_dir / f"{reconstruction_name}_report.json"
        save_reconstruction_report(result, report_file)
        
        # Success/failure exit codes
        if result.success:
            logger.info(f"Reconstruction completed successfully! Results in: {output_dir}")
            sys.exit(0)
        else:
            logger.error(f"Reconstruction failed: {result.error_message}")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Reconstruction interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)




if __name__ == "__main__":
    main()