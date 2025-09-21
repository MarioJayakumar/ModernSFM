#!/usr/bin/env python3
"""
Simple test script to validate the full reconstruction pipeline.
"""

import sys
from pathlib import Path
import logging

# Add the main directory to the path
sys.path.append(str(Path(__file__).parent))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pipeline():
    """Test the complete reconstruction pipeline."""
    try:
        # Load configuration
        from omegaconf import OmegaConf
        config = OmegaConf.load("config/base_config.yaml")
        
        # Import pipeline
        from main.pipeline.full_reconstruction import FullReconstructionPipeline
        
        # Test with small dataset first
        input_dir = Path("data/test_data")
        output_dir = Path("outputs/pipeline_test")
        
        logger.info(f"Testing pipeline with images from: {input_dir}")
        logger.info(f"Output will be saved to: {output_dir}")
        
        # Initialize pipeline
        pipeline = FullReconstructionPipeline(config)
        
        # Run reconstruction
        result = pipeline.reconstruct(
            image_dir=input_dir,
            output_dir=output_dir,
            name="pipeline_test"
        )
        
        # Print results
        if result.success:
            logger.info("✅ Pipeline test PASSED!")
            logger.info(f"Reconstructed {result.num_points_triangulated} 3D points from {result.num_images} images")
            logger.info(f"Total time: {result.total_time:.2f}s")
            if result.bundle_adjustment_result:
                ba = result.bundle_adjustment_result
                logger.info(f"Bundle adjustment: {ba.initial_rmse:.3f} → {ba.final_rmse:.3f} RMSE")
        else:
            logger.error("❌ Pipeline test FAILED!")
            logger.error(f"Error: {result.error_message}")
            
        return result.success
        
    except Exception as e:
        logger.error(f"Pipeline test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)