"""
Feature matching using SuperGlue and other modern deep learning methods.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from tqdm import tqdm

try:
    from lightglue import LightGlue
    LIGHTGLUE_AVAILABLE = True
except ImportError:
    LIGHTGLUE_AVAILABLE = False
    raise ImportError("LightGlue not available. Install with: pip install lightglue")

logger = logging.getLogger(__name__)


class FeatureMatcher:
    """
    Feature matching using modern deep learning methods.
    Currently supports SuperGlue with extensible architecture for future matchers.
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        self.matcher_type = config.feature_matching.matcher_type
        self.match_threshold = config.feature_matching.match_threshold
        self.max_matches = config.feature_matching.max_matches
        self.cross_check = config.feature_matching.cross_check
        
        # Enhanced filtering parameters
        self.confidence_weighted_filtering = getattr(config.feature_matching, 'confidence_weighted_filtering', False)
        self.multi_model_ransac = getattr(config.feature_matching, 'multi_model_ransac', False)
        self.adaptive_thresholds = getattr(config.feature_matching, 'adaptive_thresholds', False)
        
        # Initialize matcher based on type
        self.matcher = self._load_matcher()
        
        logger.info(f"FeatureMatcher initialized with {self.matcher_type} on {self.device}")
        if self.confidence_weighted_filtering:
            logger.info("Confidence-weighted filtering enabled")
        if self.multi_model_ransac:
            logger.info("Multi-model RANSAC enabled")
    
    def _load_matcher(self):
        """
        Factory method to load different matcher types.
        Currently supports LightGlue, extensible for future matchers.
        """
        if self.matcher_type.lower() == "lightglue":
            return self._load_lightglue()
        elif self.matcher_type.lower() == "superglue":
            # Future implementation - would need separate SuperGlue package
            raise NotImplementedError("SuperGlue matcher not yet implemented")
        elif self.matcher_type.lower() == "loftr":
            # Future implementation
            raise NotImplementedError("LoFTR matcher not yet implemented")
        else:
            raise ValueError(f"Unknown matcher type: {self.matcher_type}")
    
    def _load_lightglue(self):
        """Load and initialize LightGlue matcher."""
        if not LIGHTGLUE_AVAILABLE:
            raise ImportError("LightGlue not available")
        
        # Initialize LightGlue with SuperPoint features
        matcher = LightGlue(features='superpoint').eval().to(self.device)
        return matcher
    
    def match_pairs(self, features1: Dict, features2: Dict) -> Dict:
        """
        Match features between two images using the configured matcher.
        
        Args:
            features1: Features from first image (from FeatureExtractor)
            features2: Features from second image (from FeatureExtractor)
            
        Returns:
            Dictionary containing matches and metadata
        """
        if self.matcher_type.lower() == "lightglue":
            return self._match_lightglue(features1, features2)
        else:
            raise NotImplementedError(f"Matching not implemented for {self.matcher_type}")
    
    def _match_lightglue(self, features1: Dict, features2: Dict) -> Dict:
        """
        Match features using LightGlue.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Dictionary containing matches and metadata
        """
        # Convert features to LightGlue format
        data = self._prepare_lightglue_input(features1, features2)
        
        # Run LightGlue inference
        with torch.no_grad():
            pred = self.matcher(data)
        
        # Extract matches
        matches = pred['matches0'][0].cpu().numpy()  # [N] array
        confidence = pred['matching_scores0'][0].cpu().numpy()  # [N] array
        
        # Filter valid matches (LightGlue uses -1 for unmatched keypoints)
        valid_mask = matches > -1
        valid_matches = matches[valid_mask]
        valid_confidence = confidence[valid_mask]
        valid_indices1 = np.where(valid_mask)[0]
        
        # Apply confidence threshold
        conf_mask = valid_confidence >= self.match_threshold
        final_matches = valid_matches[conf_mask]
        final_confidence = valid_confidence[conf_mask]
        final_indices1 = valid_indices1[conf_mask]
        
        # Limit number of matches if specified
        if self.max_matches > 0 and len(final_matches) > self.max_matches:
            # Sort by confidence and take top matches
            sort_indices = np.argsort(final_confidence)[::-1][:self.max_matches]
            final_matches = final_matches[sort_indices]
            final_confidence = final_confidence[sort_indices]
            final_indices1 = final_indices1[sort_indices]
        
        # Create match pairs
        match_indices = np.column_stack([final_indices1, final_matches])
        
        # Get matched keypoints
        kpts1 = features1['keypoints'][final_indices1]
        kpts2 = features2['keypoints'][final_matches]
        
        return {
            'matches': match_indices,  # [N, 2] array of (idx1, idx2)
            'confidence': final_confidence,  # [N] array
            'keypoints1': kpts1,  # [N, 2] array
            'keypoints2': kpts2,  # [N, 2] array
            'num_matches': len(match_indices),
            'matcher_type': 'lightglue',
            'image_path1': features1['image_path'],
            'image_path2': features2['image_path']
        }
    
    def _prepare_lightglue_input(self, features1: Dict, features2: Dict) -> Dict:
        """
        Prepare input data for LightGlue inference.
        
        Args:
            features1: Features from first image
            features2: Features from second image
            
        Returns:
            Dictionary formatted for LightGlue
        """
        # Convert to tensors and move to device
        kpts1 = torch.from_numpy(features1['keypoints']).float().to(self.device)
        kpts2 = torch.from_numpy(features2['keypoints']).float().to(self.device)
        desc1 = torch.from_numpy(features1['descriptors']).float().to(self.device)
        desc2 = torch.from_numpy(features2['descriptors']).float().to(self.device)
        
        # Add batch dimension and prepare nested structure for LightGlue
        data = {
            'image0': {
                'keypoints': kpts1.unsqueeze(0),  # [1, N, 2]
                'descriptors': desc1.unsqueeze(0),  # [1, N, D]
                'image_size': torch.tensor([features1['image_size']], device=self.device).float()  # [1, 2]
            },
            'image1': {
                'keypoints': kpts2.unsqueeze(0),  # [1, M, 2]
                'descriptors': desc2.unsqueeze(0),  # [1, M, D]
                'image_size': torch.tensor([features2['image_size']], device=self.device).float()  # [1, 2]
            }
        }
        
        return data
    
    def match_all_pairs(self, all_features: List[Dict]) -> Dict[Tuple[int, int], Dict]:
        """
        Match features across all pairs of images efficiently.
        
        Args:
            all_features: List of feature dictionaries from FeatureExtractor
            
        Returns:
            Dictionary mapping (i, j) pairs to match results
        """
        if len(all_features) < 2:
            logger.warning("Need at least 2 images for pairwise matching")
            return {}
        
        matches_dict = {}
        total_pairs = len(all_features) * (len(all_features) - 1) // 2
        
        logger.info(f"Matching {total_pairs} image pairs")
        
        with tqdm(total=total_pairs, desc="Matching pairs") as pbar:
            for i in range(len(all_features)):
                for j in range(i + 1, len(all_features)):
                    try:
                        matches = self.match_pairs(all_features[i], all_features[j])
                        matches_dict[(i, j)] = matches
                        
                        pbar.set_postfix({
                            'pair': f"{i}-{j}",
                            'matches': matches['num_matches']
                        })
                        
                    except Exception as e:
                        logger.error(f"Failed to match pair ({i}, {j}): {e}")
                        matches_dict[(i, j)] = {
                            'matches': np.array([]),
                            'confidence': np.array([]),
                            'keypoints1': np.array([]),
                            'keypoints2': np.array([]),
                            'num_matches': 0,
                            'matcher_type': self.matcher_type,
                            'error': str(e)
                        }
                    
                    pbar.update(1)
        
        total_matches = sum(m['num_matches'] for m in matches_dict.values())
        logger.info(f"Total matches found: {total_matches}")
        
        return matches_dict
    
    def filter_matches(self, matches: Dict, method: str = "fundamental") -> Dict:
        """
        Filter matches using enhanced geometric verification with confidence weighting and multi-model RANSAC.
        
        Args:
            matches: Match dictionary from match_pairs
            method: Geometric verification method ("fundamental", "homography", "essential", "auto")
            
        Returns:
            Filtered match dictionary
        """
        if matches['num_matches'] < 8:
            logger.warning("Not enough matches for geometric verification")
            return matches
        
        kpts1 = matches['keypoints1']
        kpts2 = matches['keypoints2']
        confidence = matches.get('confidence', np.ones(len(kpts1)))
        
        # Use enhanced filtering if enabled
        if self.multi_model_ransac and method == "fundamental":
            inlier_mask, best_method = self._filter_multi_model_ransac(kpts1, kpts2, confidence)
            method = best_method
        elif self.confidence_weighted_filtering:
            inlier_mask = self._filter_confidence_weighted(kpts1, kpts2, confidence, method)
        else:
            # Standard filtering
            if method == "fundamental":
                inlier_mask = self._filter_fundamental_matrix(kpts1, kpts2)
            elif method == "homography":
                inlier_mask = self._filter_homography(kpts1, kpts2)
            elif method == "essential":
                logger.warning("Essential matrix filtering requires camera intrinsics")
                return matches
            else:
                raise ValueError(f"Unknown filtering method: {method}")
        
        # Apply inlier mask
        filtered_matches = {
            'matches': matches['matches'][inlier_mask],
            'confidence': matches['confidence'][inlier_mask],
            'keypoints1': matches['keypoints1'][inlier_mask],
            'keypoints2': matches['keypoints2'][inlier_mask],
            'num_matches': np.sum(inlier_mask),
            'matcher_type': matches['matcher_type'],
            'image_path1': matches['image_path1'],
            'image_path2': matches['image_path2'],
            'geometric_filter': method,
            'inlier_ratio': np.sum(inlier_mask) / len(inlier_mask)
        }
        
        logger.info(f"Enhanced geometric filtering ({method}): {matches['num_matches']} → {filtered_matches['num_matches']} matches")
        
        return filtered_matches
    
    def apply_spatial_distribution_to_matches(self, matches: Dict, grid_size: int = 8, 
                                            matches_per_cell: int = 8) -> Dict:
        """
        Apply spatial distribution to matches to ensure they are spread across the image.
        
        Args:
            matches: Match dictionary from match_pairs or filter_matches
            grid_size: Size of spatial grid (e.g., 8 for 8x8 grid)
            matches_per_cell: Target number of matches per grid cell
            
        Returns:
            Spatially distributed match dictionary
        """
        if matches['num_matches'] == 0:
            return matches
        
        # Get image dimensions from the first image (assuming both images have similar preprocessing)
        # We'll use the keypoints from image1 for spatial distribution
        kpts1 = matches['keypoints1']
        kpts2 = matches['keypoints2']
        confidence = matches.get('confidence', np.ones(len(kpts1)))
        
        # Estimate image size from keypoint bounds (since we don't have direct access to image size)
        if len(kpts1) > 0:
            max_x = np.max(kpts1[:, 0])
            max_y = np.max(kpts1[:, 1])
            # Add some padding to ensure we don't miss edge features
            image_w = max_x * 1.1
            image_h = max_y * 1.1
        else:
            return matches
        
        grid_w = image_w / grid_size
        grid_h = image_h / grid_size
        
        selected_indices = []
        empty_cells = []
        
        # First pass: collect matches from each cell based on image1 keypoints
        for i in range(grid_size):
            for j in range(grid_size):
                # Define grid cell boundaries
                y_min = i * grid_h
                y_max = (i + 1) * grid_h
                x_min = j * grid_w
                x_max = (j + 1) * grid_w
                
                # Find matches where the first keypoint falls in this cell
                in_cell = (
                    (kpts1[:, 0] >= x_min) & (kpts1[:, 0] < x_max) &
                    (kpts1[:, 1] >= y_min) & (kpts1[:, 1] < y_max)
                )
                
                cell_indices = np.where(in_cell)[0]
                
                if len(cell_indices) > 0:
                    # Sort by confidence and take top matches for this cell
                    cell_confidence = confidence[cell_indices]
                    sorted_cell_indices = cell_indices[np.argsort(cell_confidence)[::-1]]
                    
                    # Take up to matches_per_cell matches from this cell
                    n_take = min(matches_per_cell, len(sorted_cell_indices))
                    selected_indices.extend(sorted_cell_indices[:n_take])
                else:
                    empty_cells.append((i, j, x_min, x_max, y_min, y_max))
        
        # Second pass: try to fill empty cells with nearby matches
        if empty_cells and len(selected_indices) < len(kpts1):
            logger.info(f"Found {len(empty_cells)} empty cells, attempting to fill with nearby matches")
            
            for i, j, x_min, x_max, y_min, y_max in empty_cells:
                # Expand search radius to find nearby matches
                search_radius = max(grid_h, grid_w) * 1.5
                
                # Find matches within expanded radius of cell center
                cell_center_x = (x_min + x_max) / 2
                cell_center_y = (y_min + y_max) / 2
                
                distances = np.sqrt((kpts1[:, 0] - cell_center_x)**2 + (kpts1[:, 1] - cell_center_y)**2)
                nearby_indices = np.where(distances <= search_radius)[0]
                
                # Filter out already selected matches
                available_indices = [idx for idx in nearby_indices if idx not in selected_indices]
                
                if available_indices:
                    # Take the best available match for this empty cell
                    available_confidence = confidence[available_indices]
                    best_idx = available_indices[np.argmax(available_confidence)]
                    selected_indices.append(best_idx)
        
        if not selected_indices:
            logger.warning("No matches selected after spatial distribution - returning original matches")
            return matches
        
        selected_indices = np.array(selected_indices)
        
        # Create spatially distributed match dictionary
        distributed_matches = {
            'matches': matches['matches'][selected_indices],
            'confidence': matches['confidence'][selected_indices],
            'keypoints1': matches['keypoints1'][selected_indices],
            'keypoints2': matches['keypoints2'][selected_indices],
            'num_matches': len(selected_indices),
            'matcher_type': matches['matcher_type'],
            'image_path1': matches['image_path1'],
            'image_path2': matches['image_path2'],
            'spatial_distribution': f"{grid_size}x{grid_size}_grid"
        }
        
        # Copy any additional fields
        for key in ['geometric_filter', 'inlier_ratio']:
            if key in matches:
                distributed_matches[key] = matches[key]
        
        # Log spatial distribution statistics
        final_count = len(selected_indices)
        cells_with_matches = grid_size * grid_size - len(empty_cells)
        logger.info(f"Match spatial distribution: {final_count} matches across {cells_with_matches}/{grid_size*grid_size} cells")
        
        return distributed_matches
    
    def _filter_fundamental_matrix(self, kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
        """Filter matches using fundamental matrix estimation."""
        try:
            _, inlier_mask = cv2.findFundamentalMat(
                kpts1, kpts2,
                cv2.FM_RANSAC,
                ransacReprojThreshold=1.0,
                confidence=0.99
            )
            return inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(kpts1), dtype=bool)
        except Exception as e:
            logger.warning(f"Fundamental matrix filtering failed: {e}")
            return np.ones(len(kpts1), dtype=bool)
    
    def _filter_homography(self, kpts1: np.ndarray, kpts2: np.ndarray) -> np.ndarray:
        """Filter matches using homography estimation."""
        try:
            _, inlier_mask = cv2.findHomography(
                kpts1, kpts2,
                cv2.RANSAC,
                ransacReprojThreshold=3.0,
                confidence=0.99
            )
            return inlier_mask.ravel().astype(bool) if inlier_mask is not None else np.ones(len(kpts1), dtype=bool)
        except Exception as e:
            logger.warning(f"Homography filtering failed: {e}")
            return np.ones(len(kpts1), dtype=bool)
    
    def _filter_multi_model_ransac(self, kpts1: np.ndarray, kpts2: np.ndarray, 
                                 confidence: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Filter matches using multi-model RANSAC (fundamental matrix vs homography).
        
        Args:
            kpts1: Keypoints from first image
            kpts2: Keypoints from second image
            confidence: Match confidence scores
            
        Returns:
            Tuple of (inlier_mask, best_method)
        """
        # Try both fundamental matrix and homography
        fund_mask = self._filter_fundamental_matrix(kpts1, kpts2)
        homo_mask = self._filter_homography(kpts1, kpts2)
        
        fund_inliers = np.sum(fund_mask)
        homo_inliers = np.sum(homo_mask)
        
        # Weight by confidence if available
        if len(confidence) > 0:
            fund_conf_score = np.sum(confidence[fund_mask]) if fund_inliers > 0 else 0
            homo_conf_score = np.sum(confidence[homo_mask]) if homo_inliers > 0 else 0
            
            # Combine inlier count and confidence score
            fund_score = fund_inliers * 0.7 + fund_conf_score * 0.3
            homo_score = homo_inliers * 0.7 + homo_conf_score * 0.3
        else:
            fund_score = fund_inliers
            homo_score = homo_inliers
        
        # Choose the better model
        if fund_score >= homo_score:
            logger.info(f"Multi-model RANSAC: Fundamental matrix selected ({fund_inliers} inliers, score: {fund_score:.2f})")
            return fund_mask, "fundamental"
        else:
            logger.info(f"Multi-model RANSAC: Homography selected ({homo_inliers} inliers, score: {homo_score:.2f})")
            return homo_mask, "homography"
    
    def _filter_confidence_weighted(self, kpts1: np.ndarray, kpts2: np.ndarray, 
                                  confidence: np.ndarray, method: str) -> np.ndarray:
        """
        Filter matches using confidence-weighted geometric verification.
        
        Args:
            kpts1: Keypoints from first image
            kpts2: Keypoints from second image
            confidence: Match confidence scores
            method: Geometric method to use
            
        Returns:
            Inlier mask
        """
        # Compute adaptive threshold based on confidence distribution
        if self.adaptive_thresholds and len(confidence) > 0:
            # Use confidence percentiles to set adaptive thresholds
            conf_median = np.median(confidence)
            conf_std = np.std(confidence)
            
            # Adaptive RANSAC threshold based on confidence
            if method == "fundamental":
                base_threshold = 1.0
                adaptive_threshold = base_threshold * (2.0 - conf_median)  # Lower threshold for high confidence
                adaptive_threshold = np.clip(adaptive_threshold, 0.5, 2.0)
            else:  # homography
                base_threshold = 3.0
                adaptive_threshold = base_threshold * (2.0 - conf_median)
                adaptive_threshold = np.clip(adaptive_threshold, 1.5, 5.0)
        else:
            adaptive_threshold = 1.0 if method == "fundamental" else 3.0
        
        # Apply geometric filtering with adaptive threshold
        try:
            if method == "fundamental":
                _, inlier_mask = cv2.findFundamentalMat(
                    kpts1, kpts2,
                    cv2.FM_RANSAC,
                    ransacReprojThreshold=adaptive_threshold,
                    confidence=0.99
                )
            elif method == "homography":
                _, inlier_mask = cv2.findHomography(
                    kpts1, kpts2,
                    cv2.RANSAC,
                    ransacReprojThreshold=adaptive_threshold,
                    confidence=0.99
                )
            else:
                raise ValueError(f"Unknown method: {method}")
            
            if inlier_mask is None:
                return np.ones(len(kpts1), dtype=bool)
            
            inlier_mask = inlier_mask.ravel().astype(bool)
            
            # Additional confidence-based filtering
            if len(confidence) > 0:
                # Keep high-confidence matches even if they're geometric outliers
                high_conf_threshold = np.percentile(confidence, 80)
                high_conf_mask = confidence >= high_conf_threshold
                
                # Combine geometric inliers with high-confidence matches
                combined_mask = inlier_mask | high_conf_mask
                
                logger.info(f"Confidence-weighted filtering: {np.sum(inlier_mask)} geometric + {np.sum(high_conf_mask)} high-conf = {np.sum(combined_mask)} total")
                return combined_mask
            else:
                return inlier_mask
                
        except Exception as e:
            logger.warning(f"Confidence-weighted filtering failed: {e}")
            return np.ones(len(kpts1), dtype=bool)
    
    def visualize_matches(self, image_path1: Union[str, Path], image_path2: Union[str, Path],
                         matches: Dict, output_path: Optional[Union[str, Path]] = None,
                         max_matches: int = 100) -> np.ndarray:
        """
        Visualize matches between two images.
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            matches: Match dictionary from match_pairs
            output_path: Optional path to save visualization
            max_matches: Maximum number of matches to visualize
            
        Returns:
            Visualization image
        """
        # Load images at original resolution
        img1 = cv2.imread(str(image_path1))
        img2 = cv2.imread(str(image_path2))
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        # Calculate scale factors to convert keypoints from preprocessed to original coordinates
        max_size = self.config.io.max_image_size
        
        # Scale factor for image 1
        original_max_dim1 = max(img1.shape[:2])
        scale_factor1 = 1.0
        if original_max_dim1 > max_size:
            scale_factor1 = original_max_dim1 / max_size
        
        # Scale factor for image 2
        original_max_dim2 = max(img2.shape[:2])
        scale_factor2 = 1.0
        if original_max_dim2 > max_size:
            scale_factor2 = original_max_dim2 / max_size
        
        # Get keypoints and scale them to original image coordinates
        kpts1 = matches['keypoints1'] * scale_factor1
        kpts2 = matches['keypoints2'] * scale_factor2
        confidence = matches.get('confidence', np.ones(len(kpts1)))
        
        if len(kpts1) > max_matches:
            # Sort by confidence and take top matches
            indices = np.argsort(confidence)[::-1][:max_matches]
            kpts1 = kpts1[indices]
            kpts2 = kpts2[indices]
            confidence = confidence[indices]
        
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        
        # Draw matches
        for i, (pt1, pt2, conf) in enumerate(zip(kpts1, kpts2, confidence)):
            # Color based on confidence (green = high, red = low)
            color = (int(255 * (1 - conf)), int(255 * conf), 0)
            
            # Draw keypoints
            cv2.circle(vis_img, (int(pt1[0]), int(pt1[1])), 3, color, -1)
            cv2.circle(vis_img, (int(pt2[0] + w1), int(pt2[1])), 3, color, -1)
            
            # Draw match line
            cv2.line(vis_img, 
                    (int(pt1[0]), int(pt1[1])), 
                    (int(pt2[0] + w1), int(pt2[1])), 
                    color, 1)
        
        # Add text info
        info_text = f"Matches: {matches['num_matches']} | {matches['matcher_type']}"
        if 'geometric_filter' in matches:
            info_text += f" | {matches['geometric_filter']} filtered"
        
        cv2.putText(vis_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), vis_img)
            logger.info(f"Match visualization saved to {output_path}")
        
        return vis_img
    
    def visualize_matches_with_all_keypoints(self, image_path1: Union[str, Path], image_path2: Union[str, Path],
                                           features1: Dict, features2: Dict, matches: Dict, 
                                           output_path: Optional[Union[str, Path]] = None,
                                           max_matches: int = 100, max_keypoints_per_image: int = 500) -> np.ndarray:
        """
        Enhanced visualization showing matched pairs overlaid on all detected keypoints.
        This provides better debugging by showing which keypoints were matched vs ignored.
        
        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            features1: All features from first image (from FeatureExtractor)
            features2: All features from second image (from FeatureExtractor)
            matches: Match dictionary from match_pairs
            output_path: Optional path to save visualization
            max_matches: Maximum number of matches to visualize
            max_keypoints_per_image: Maximum number of keypoints to show per image
            
        Returns:
            Enhanced visualization image
        """
        # Load images at original resolution
        img1 = cv2.imread(str(image_path1))
        img2 = cv2.imread(str(image_path2))
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        # Calculate scale factors to convert keypoints from preprocessed to original coordinates
        max_size = self.config.io.max_image_size
        
        # Scale factor for image 1
        original_max_dim1 = max(img1.shape[:2])
        scale_factor1 = 1.0
        if original_max_dim1 > max_size:
            scale_factor1 = original_max_dim1 / max_size
        
        # Scale factor for image 2
        original_max_dim2 = max(img2.shape[:2])
        scale_factor2 = 1.0
        if original_max_dim2 > max_size:
            scale_factor2 = original_max_dim2 / max_size
        
        # Create side-by-side image
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        h = max(h1, h2)
        w = w1 + w2
        
        vis_img = np.zeros((h, w, 3), dtype=np.uint8)
        vis_img[:h1, :w1] = img1
        vis_img[:h2, w1:w1+w2] = img2
        
        # Step 1: Draw ALL detected keypoints in subtle gray
        all_kpts1 = features1['keypoints'] * scale_factor1
        all_kpts2 = features2['keypoints'] * scale_factor2
        all_scores1 = features1.get('scores', np.ones(len(all_kpts1)))
        all_scores2 = features2.get('scores', np.ones(len(all_kpts2)))
        
        # Limit keypoints for visualization if too many
        if len(all_kpts1) > max_keypoints_per_image:
            # Sort by score and take top keypoints
            indices1 = np.argsort(all_scores1)[::-1][:max_keypoints_per_image]
            all_kpts1 = all_kpts1[indices1]
            all_scores1 = all_scores1[indices1]
        
        if len(all_kpts2) > max_keypoints_per_image:
            indices2 = np.argsort(all_scores2)[::-1][:max_keypoints_per_image]
            all_kpts2 = all_kpts2[indices2]
            all_scores2 = all_scores2[indices2]
        
        # Draw all keypoints in subtle gray (unmatched keypoints)
        for kp, score in zip(all_kpts1, all_scores1):
            # Gray color with intensity based on keypoint score
            intensity = int(100 + 100 * score)  # Range: 100-200 for visibility
            color = (intensity, intensity, intensity)
            cv2.circle(vis_img, (int(kp[0]), int(kp[1])), 2, color, -1)
        
        for kp, score in zip(all_kpts2, all_scores2):
            intensity = int(100 + 100 * score)
            color = (intensity, intensity, intensity)
            cv2.circle(vis_img, (int(kp[0] + w1), int(kp[1])), 2, color, -1)
        
        # Step 2: Draw matched keypoints and connections in bright colors
        if matches['num_matches'] > 0:
            # Get matched keypoints and scale them to original image coordinates
            matched_kpts1 = matches['keypoints1'] * scale_factor1
            matched_kpts2 = matches['keypoints2'] * scale_factor2
            confidence = matches.get('confidence', np.ones(len(matched_kpts1)))
            
            # Limit matches for visualization
            if len(matched_kpts1) > max_matches:
                # Sort by confidence and take top matches
                indices = np.argsort(confidence)[::-1][:max_matches]
                matched_kpts1 = matched_kpts1[indices]
                matched_kpts2 = matched_kpts2[indices]
                confidence = confidence[indices]
            
            # Draw match lines first (so they appear behind keypoints)
            for pt1, pt2, conf in zip(matched_kpts1, matched_kpts2, confidence):
                # Color based on confidence (green = high, red = low)
                color = (int(255 * (1 - conf)), int(255 * conf), 0)
                cv2.line(vis_img, 
                        (int(pt1[0]), int(pt1[1])), 
                        (int(pt2[0] + w1), int(pt2[1])), 
                        color, 2)  # Thicker lines for matches
            
            # Draw matched keypoints on top with bright colors and larger circles
            for pt1, pt2, conf in zip(matched_kpts1, matched_kpts2, confidence):
                # Color based on confidence (green = high, red = low)
                color = (int(255 * (1 - conf)), int(255 * conf), 0)
                
                # Draw larger, brighter circles for matched keypoints
                cv2.circle(vis_img, (int(pt1[0]), int(pt1[1])), 4, color, -1)
                cv2.circle(vis_img, (int(pt2[0] + w1), int(pt2[1])), 4, color, -1)
                
                # Add white border for better visibility
                cv2.circle(vis_img, (int(pt1[0]), int(pt1[1])), 5, (255, 255, 255), 1)
                cv2.circle(vis_img, (int(pt2[0] + w1), int(pt2[1])), 5, (255, 255, 255), 1)
        
        # Step 3: Add comprehensive text info
        info_lines = [
            f"Total keypoints: {len(all_kpts1)} | {len(all_kpts2)}",
            f"Matches: {matches['num_matches']} | {matches['matcher_type']}",
        ]
        
        if 'geometric_filter' in matches:
            info_lines.append(f"Filter: {matches['geometric_filter']}")
            if 'inlier_ratio' in matches:
                info_lines.append(f"Inlier ratio: {matches['inlier_ratio']:.3f}")
        
        if 'spatial_distribution' in matches:
            info_lines.append(f"Spatial: {matches['spatial_distribution']}")
        
        # Draw text info with background for better readability
        y_offset = 25
        for line in info_lines:
            # Draw background rectangle
            text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(vis_img, (5, y_offset - 20), (text_size[0] + 10, y_offset + 5), (0, 0, 0), -1)
            # Draw text
            cv2.putText(vis_img, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
        
        # Add legend
        legend_y = h - 80
        cv2.rectangle(vis_img, (5, legend_y - 5), (300, h - 5), (0, 0, 0), -1)
        cv2.putText(vis_img, "Legend:", (10, legend_y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(vis_img, (20, legend_y + 35), 2, (150, 150, 150), -1)
        cv2.putText(vis_img, "All detected keypoints", (35, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(vis_img, (20, legend_y + 55), 4, (0, 255, 0), -1)
        cv2.circle(vis_img, (20, legend_y + 55), 5, (255, 255, 255), 1)
        cv2.putText(vis_img, "Matched keypoints (green=high conf)", (35, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save if requested
        if output_path:
            cv2.imwrite(str(output_path), vis_img)
            logger.info(f"Enhanced match visualization saved to {output_path}")
        
        return vis_img
