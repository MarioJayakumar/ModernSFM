"""
Feature extraction using LightGlue SuperPoint.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Sequence
import logging
from tqdm import tqdm

try:
    from lightglue import SuperPoint
    from lightglue.utils import load_image
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    raise ImportError("LightGlue not available. Install with: pip install lightglue")

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Feature extraction using LightGlue SuperPoint.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.max_keypoints = config.feature_extractor.max_keypoints
        self.detection_threshold = config.feature_extractor.detection_threshold
        
        # Multi-scale and spatial distribution parameters
        self.multi_scale = getattr(config.feature_extractor, 'multi_scale', False)
        self.scales = getattr(config.feature_extractor, 'scales', [1.0])
        self.spatial_grid = getattr(config.feature_extractor, 'spatial_grid', 0)
        self.features_per_cell = getattr(config.feature_extractor, 'features_per_cell', 0)
        self.min_features_per_cell = getattr(config.feature_extractor, 'min_features_per_cell', 0)
        self.enforce_spatial_distribution = getattr(config.feature_extractor, 'enforce_spatial_distribution', False)
        
        # Initialize SuperPoint extractor
        self.extractor = SuperPoint(max_num_keypoints=self.max_keypoints).eval().to(self.device)
        
        logger.info(f"FeatureExtractor initialized with SuperPoint on {self.device}")
        if self.multi_scale:
            logger.info(f"Multi-scale extraction enabled with scales: {self.scales}")
        if self.spatial_grid > 0:
            logger.info(f"Spatial distribution enabled with {self.spatial_grid}x{self.spatial_grid} grid")
    
    def _preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Preprocess a single image for feature extraction.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor [C, H, W]
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image using LightGlue utilities
        image = load_image(image_path)
        
        # Convert RGB to grayscale if needed (SuperPoint expects 1 channel)
        if image.shape[0] == 3:  # RGB image
            # Convert RGB to grayscale using standard luminance weights
            image = 0.299 * image[0] + 0.587 * image[1] + 0.114 * image[2]
            image = image.unsqueeze(0)  # Add channel dimension back
        
        # Resize if needed
        max_size = self.config.io.max_image_size
        if max(image.shape[-2:]) > max_size:
            scale = max_size / max(image.shape[-2:])
            new_size = (int(image.shape[-2] * scale), int(image.shape[-1] * scale))
            image = torch.nn.functional.interpolate(
                image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False
            ).squeeze(0)
        
        return image
    
    def extract_features_batch(self, image_paths: Sequence[Union[str, Path]]) -> List[Dict]:
        """
        Extract features from multiple images. 
        Note: Due to variable keypoint counts, we process images individually rather than true batching.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            List of feature dictionaries
        """
        if not image_paths:
            return []
        
        features_list = []
        
        logger.info(f"Extracting features from {len(image_paths)} images")
        
        for image_path in tqdm(image_paths, desc="Extracting features"):
            try:
                # Use single image extraction for each image to avoid tensor size mismatch
                features = self.extract_features_single(image_path)
                features_list.append(features)
            except Exception as e:
                logger.error(f"Failed to extract features from {image_path}: {e}")
                continue
        
        logger.info(f"Successfully extracted features from {len(features_list)}/{len(image_paths)} images")
        return features_list
    
    def _extract_multiscale_features(self, image: torch.Tensor, image_path: str) -> Dict:
        """
        Extract features at multiple scales and combine them.
        
        Args:
            image: Preprocessed image tensor [C, H, W]
            image_path: Path to the image file
            
        Returns:
            Combined features from all scales
        """
        all_keypoints = []
        all_descriptors = []
        all_scores = []
        original_size = image.shape[-2:]
        
        for scale in self.scales:
            if scale != 1.0:
                # Resize image for this scale
                new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
                scaled_image = torch.nn.functional.interpolate(
                    image.unsqueeze(0), size=new_size, mode='bilinear', align_corners=False
                ).squeeze(0)
            else:
                scaled_image = image
            
            # Extract features at this scale
            with torch.no_grad():
                features = self.extractor({'image': scaled_image.unsqueeze(0).to(self.device)})
            
            keypoints = features['keypoints'][0].cpu().numpy()
            descriptors = features['descriptors'][0].cpu().numpy()
            scores = features['keypoint_scores'][0].cpu().numpy()
            
            # Scale keypoints back to original image coordinates
            if scale != 1.0:
                keypoints = keypoints / scale
            
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
            all_scores.append(scores)
        
        # Combine features from all scales
        combined_keypoints = np.vstack(all_keypoints) if all_keypoints else np.array([])
        combined_descriptors = np.vstack(all_descriptors) if all_descriptors else np.array([])
        combined_scores = np.concatenate(all_scores) if all_scores else np.array([])
        
        # Remove duplicates and apply NMS
        if len(combined_keypoints) > 0:
            combined_keypoints, combined_descriptors, combined_scores = self._remove_duplicate_features(
                combined_keypoints, combined_descriptors, combined_scores
            )
        
        return {
            'keypoints': combined_keypoints,
            'descriptors': combined_descriptors,
            'scores': combined_scores,
            'image_path': image_path,
            'image_size': original_size,
            'num_features': len(combined_keypoints),
            'extractor_type': 'superpoint_multiscale'
        }
    
    def _remove_duplicate_features(self, keypoints: np.ndarray, descriptors: np.ndarray, 
                                 scores: np.ndarray, radius: float = 3.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Remove duplicate features using non-maximum suppression.
        
        Args:
            keypoints: Array of keypoints [N, 2]
            descriptors: Array of descriptors [N, D]
            scores: Array of scores [N]
            radius: NMS radius in pixels
            
        Returns:
            Filtered keypoints, descriptors, and scores
        """
        if len(keypoints) == 0:
            return keypoints, descriptors, scores
        
        # Sort by score (highest first)
        sorted_indices = np.argsort(scores)[::-1]
        
        keep_indices = []
        for i in sorted_indices:
            kp = keypoints[i]
            
            # Check if this keypoint is too close to any already kept keypoint
            too_close = False
            for j in keep_indices:
                dist = np.linalg.norm(kp - keypoints[j])
                if dist < radius:
                    too_close = True
                    break
            
            if not too_close:
                keep_indices.append(i)
        
        return keypoints[keep_indices], descriptors[keep_indices], scores[keep_indices]
    
    def _apply_spatial_distribution(self, keypoints: np.ndarray, descriptors: np.ndarray, 
                                  scores: np.ndarray, image_size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply aggressive spatial distribution to ensure features are spread across the entire image.
        
        Args:
            keypoints: Array of keypoints [N, 2]
            descriptors: Array of descriptors [N, D]
            scores: Array of scores [N]
            image_size: (height, width) of the image
            
        Returns:
            Spatially distributed keypoints, descriptors, and scores
        """
        if len(keypoints) == 0 or self.spatial_grid <= 0:
            return keypoints, descriptors, scores
        
        h, w = image_size
        grid_h = h / self.spatial_grid  # Use float division for more precise boundaries
        grid_w = w / self.spatial_grid
        
        selected_indices = []
        empty_cells = []
        
        # First pass: collect features from each cell
        for i in range(self.spatial_grid):
            for j in range(self.spatial_grid):
                # Define grid cell boundaries with float precision
                y_min = i * grid_h
                y_max = (i + 1) * grid_h
                x_min = j * grid_w
                x_max = (j + 1) * grid_w
                
                # Find keypoints in this cell
                in_cell = (
                    (keypoints[:, 0] >= x_min) & (keypoints[:, 0] < x_max) &
                    (keypoints[:, 1] >= y_min) & (keypoints[:, 1] < y_max)
                )
                
                cell_indices = np.where(in_cell)[0]
                
                if len(cell_indices) > 0:
                    # Sort by score and take top features for this cell
                    cell_scores = scores[cell_indices]
                    sorted_cell_indices = cell_indices[np.argsort(cell_scores)[::-1]]
                    
                    # Take up to features_per_cell features from this cell
                    n_take = min(self.features_per_cell, len(sorted_cell_indices))
                    selected_indices.extend(sorted_cell_indices[:n_take])
                else:
                    # Track empty cells for potential filling
                    empty_cells.append((i, j, x_min, x_max, y_min, y_max))
        
        # Second pass: try to fill empty cells with nearby features if enforce_spatial_distribution is True
        if self.enforce_spatial_distribution and empty_cells:
            logger.info(f"Found {len(empty_cells)} empty cells, attempting to fill with nearby features")
            
            for i, j, x_min, x_max, y_min, y_max in empty_cells:
                # Expand search radius to find nearby features
                search_radius = max(grid_h, grid_w) * 1.5  # Search in neighboring cells
                
                # Find features within expanded radius of cell center
                cell_center_x = (x_min + x_max) / 2
                cell_center_y = (y_min + y_max) / 2
                
                distances = np.sqrt((keypoints[:, 0] - cell_center_x)**2 + (keypoints[:, 1] - cell_center_y)**2)
                nearby_indices = np.where(distances <= search_radius)[0]
                
                # Filter out already selected features
                available_indices = [idx for idx in nearby_indices if idx not in selected_indices]
                
                if available_indices:
                    # Take the best available feature for this empty cell
                    available_scores = scores[available_indices]
                    best_idx = available_indices[np.argmax(available_scores)]
                    selected_indices.append(best_idx)
                    logger.debug(f"Filled empty cell ({i},{j}) with feature at distance {distances[best_idx]:.1f}")
        
        # Third pass: ensure minimum features per cell if specified
        if self.min_features_per_cell > 0:
            current_count = len(selected_indices)
            target_count = self.spatial_grid * self.spatial_grid * self.min_features_per_cell
            
            if current_count < target_count:
                # Add more features to reach minimum
                remaining_indices = [i for i in range(len(keypoints)) if i not in selected_indices]
                if remaining_indices:
                    remaining_scores = scores[remaining_indices]
                    sorted_remaining = [remaining_indices[i] for i in np.argsort(remaining_scores)[::-1]]
                    
                    n_add = min(target_count - current_count, len(sorted_remaining))
                    selected_indices.extend(sorted_remaining[:n_add])
                    logger.info(f"Added {n_add} features to reach minimum spatial distribution")
        
        if not selected_indices:
            logger.warning("No features selected after spatial distribution - returning original features")
            return keypoints, descriptors, scores
        
        selected_indices = np.array(selected_indices)
        
        # Log spatial distribution statistics
        final_count = len(selected_indices)
        cells_with_features = self.spatial_grid * self.spatial_grid - len(empty_cells)
        logger.info(f"Spatial distribution: {final_count} features across {cells_with_features}/{self.spatial_grid*self.spatial_grid} cells")
        
        return keypoints[selected_indices], descriptors[selected_indices], scores[selected_indices]
    
    def extract_features_single(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract features from a single image using SuperPoint with multi-scale and spatial distribution.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing keypoints, descriptors, and metadata
        """
        # Preprocess image
        image = self._preprocess_image(image_path)
        
        # Extract features (multi-scale if enabled)
        if self.multi_scale and len(self.scales) > 1:
            features = self._extract_multiscale_features(image, str(image_path))
        else:
            # Single scale extraction
            with torch.no_grad():
                batch_features = self.extractor({'image': image.unsqueeze(0).to(self.device)})
            
            keypoints = batch_features['keypoints'][0].cpu().numpy()
            descriptors = batch_features['descriptors'][0].cpu().numpy()
            scores = batch_features['keypoint_scores'][0].cpu().numpy()
            
            features = {
                'keypoints': keypoints,
                'descriptors': descriptors,
                'scores': scores,
                'image_path': str(image_path),
                'image_size': image.shape[-2:],
                'num_features': len(keypoints),
                'extractor_type': 'superpoint'
            }
        
        # Apply spatial distribution if enabled
        if self.spatial_grid > 0 and self.features_per_cell > 0:
            features['keypoints'], features['descriptors'], features['scores'] = self._apply_spatial_distribution(
                features['keypoints'], features['descriptors'], features['scores'], features['image_size']
            )
            features['num_features'] = len(features['keypoints'])
            features['extractor_type'] += '_spatial'
        
        return features
    
    def visualize_features(self, image_path: Union[str, Path], features: Dict, 
                          output_path: Optional[Union[str, Path]] = None) -> np.ndarray:
        """
        Visualize extracted features on the image.
        
        Args:
            image_path: Path to the original image
            features: Features dictionary from extract_features_single
            output_path: Optional path to save the visualization
            
        Returns:
            Image with features drawn
        """
        # Load image at original size
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Calculate scale factor used during preprocessing
        max_size = self.config.io.max_image_size
        original_max_dim = max(image.shape[:2])
        scale_factor = 1.0
        
        if original_max_dim > max_size:
            scale_factor = original_max_dim / max_size
        
        # Scale up keypoints to original image coordinates
        keypoints = features['keypoints'] * scale_factor
        scores = features.get('scores', np.ones(len(keypoints)))
        
        for i, (kp, score) in enumerate(zip(keypoints, scores)):
            x, y = int(kp[0]), int(kp[1])
            # Color based on score (red = high, blue = low)
            color = (int(255 * (1 - score)), 0, int(255 * score))
            cv2.circle(image, (x, y), 3, color, -1)
        
        # Add text info
        info_text = f"Features: {features['num_features']} | SuperPoint"
        cv2.putText(image, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), image)
            logger.info(f"Feature visualization saved to {output_path}")
        
        return image


def create_feature_extractor(config):
    """Factory function to create feature extractor."""
    return FeatureExtractor(config)


def load_images_from_directory(directory: Union[str, Path], 
                             extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Load all image files from a directory.
    
    Args:
        directory: Path to directory containing images
        extensions: List of file extensions to include
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
    
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    
    image_paths = []
    for ext in extensions:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))
    
    return sorted(image_paths)
