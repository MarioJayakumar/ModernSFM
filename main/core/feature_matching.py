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
        
        # Initialize matcher based on type
        self.matcher = self._load_matcher()
        
        logger.info(f"FeatureMatcher initialized with {self.matcher_type} on {self.device}")
    
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
        Filter matches using geometric verification.
        
        Args:
            matches: Match dictionary from match_pairs
            method: Geometric verification method ("fundamental", "homography", "essential")
            
        Returns:
            Filtered match dictionary
        """
        if matches['num_matches'] < 8:
            logger.warning("Not enough matches for geometric verification")
            return matches
        
        kpts1 = matches['keypoints1']
        kpts2 = matches['keypoints2']
        
        if method == "fundamental":
            inlier_mask = self._filter_fundamental_matrix(kpts1, kpts2)
        elif method == "homography":
            inlier_mask = self._filter_homography(kpts1, kpts2)
        elif method == "essential":
            # Would need camera intrinsics for essential matrix
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
        
        logger.info(f"Geometric filtering ({method}): {matches['num_matches']} → {filtered_matches['num_matches']} matches")
        
        return filtered_matches
    
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
        # Load images
        img1 = cv2.imread(str(image_path1))
        img2 = cv2.imread(str(image_path2))
        
        if img1 is None or img2 is None:
            raise ValueError("Could not load one or both images")
        
        # Get keypoints and limit number for visualization
        kpts1 = matches['keypoints1']
        kpts2 = matches['keypoints2']
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
