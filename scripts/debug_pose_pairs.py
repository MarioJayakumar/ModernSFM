#!/usr/bin/env python3
"""
Diagnostic utility for inspecting per-pair pose estimation metrics.

For each image pair, this script reports:
  * Raw and filtered match counts from LightGlue.
  * COLMAP and OpenCV relative pose estimates (rotation angle, positive depth count, inliers).
  * Ground-truth relative rotation angle when synthetic ground truth is available.

Usage:
    python -m scripts.debug_pose_pairs \
        --images data/synthetic/simple_cube_20251025_174558/images \
        --ground_truth data/synthetic/simple_cube_20251025_174558/ground_truth.json \
        --config config/base_config.yaml \
        [--output pose_diagnostics.json]
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from omegaconf import OmegaConf

from main.core.feature_extraction import FeatureExtractor
from main.core.feature_matching import FeatureMatcher
from main.core.pose_estimation import PoseEstimator
from main.synthetic.ground_truth import load_ground_truth


def to_namespace(obj: Any) -> Any:
    """Recursively convert dictionaries into SimpleNamespace structures."""
    if isinstance(obj, dict):
        from types import SimpleNamespace

        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def rotation_angle_deg(R: np.ndarray) -> float:
    """Return the absolute rotation angle represented by rotation matrix."""
    try:
        val = (np.trace(R) - 1.0) / 2.0
        val = float(np.clip(val, -1.0, 1.0))
        return float(math.degrees(math.acos(val)))
    except Exception:
        return float("nan")


def evaluate_solver(name: str, solver_fn) -> Dict[str, Any]:
    """Run a pose solver and capture diagnostics."""
    try:
        result = solver_fn()
    except Exception as err:  # pylint: disable=broad-except
        return {"success": False, "error": str(err)}

    if not result or not result.get("success"):
        return {"success": False, "error": "solver returned failure"}

    return {
        "success": True,
        "num_inliers": int(result.get("num_inliers", 0)),
        "inlier_ratio": float(result.get("inlier_ratio", 0.0)),
        "positive_depth": int(result.get("positive_depth_count", 0)),
        "rotation_angle_deg": float(result.get("rotation_angle_deg", rotation_angle_deg(result.get("R")))),
    }


def compute_ground_truth_metrics(
    camera_i: Any, camera_j: Any
) -> Dict[str, float]:
    """Compute relative rotation angle from ground truth extrinsics."""
    if hasattr(camera_i, "rotation"):
        R_i = np.array(camera_i.rotation)
        R_j = np.array(camera_j.rotation)
    else:
        R_i = np.array(camera_i["extrinsics"]["rotation"])
        R_j = np.array(camera_j["extrinsics"]["rotation"])

    # Relative rotation from camera i to j (world-to-camera matrices)
    R_rel = R_j @ R_i.T

    return {
        "rotation_angle_deg": rotation_angle_deg(R_rel),
    }


def gather_pose_diagnostics(
    features: List[Dict[str, Any]],
    matches_dict: Dict[Tuple[int, int], Dict[str, Any]],
    pose_estimator: PoseEstimator,
    feature_matcher: FeatureMatcher,
    intrinsics: np.ndarray,
    cameras_gt: Dict[str, Any] = None,
) -> List[Dict[str, Any]]:
    """Collect diagnostic metrics for each image pair."""
    diagnostics = []

    for (i, j), match_data in sorted(matches_dict.items()):
        if match_data is None or "matches" not in match_data or len(match_data["matches"]) < 8:
            continue

        entry: Dict[str, Any] = {
            "pair": f"{i}-{j}",
            "raw_match_count": int(match_data.get("num_matches", len(match_data["matches"]))),
        }

        # Ground-truth metrics if available
        if cameras_gt:
            cam_i = cameras_gt.get(f"camera_{i:03d}")
            cam_j = cameras_gt.get(f"camera_{j:03d}")
            if cam_i and cam_j:
                entry["ground_truth"] = compute_ground_truth_metrics(cam_i, cam_j)

        kpts1_raw = features[i]["keypoints"][match_data["matches"][:, 0]]
        kpts2_raw = features[j]["keypoints"][match_data["matches"][:, 1]]

        # Raw solver outputs
        entry["raw"] = {
            "colmap": evaluate_solver(
                "colmap",
                lambda: pose_estimator._estimate_pose_colmap(  # pylint: disable=protected-access
                    kpts1_raw, kpts2_raw, intrinsics, intrinsics
                ),
            ),
            "opencv": evaluate_solver(
                "opencv",
                lambda: pose_estimator._estimate_pose_opencv(  # pylint: disable=protected-access
                    kpts1_raw, kpts2_raw, intrinsics, intrinsics
                ),
            ),
        }

        # Geometrically filtered matches
        filtered = match_data
        if match_data.get("geometric_filter") != "fundamental":
            try:
                filtered = feature_matcher.filter_matches(match_data, method="fundamental")
            except Exception:  # pylint: disable=broad-except
                filtered = match_data

        filtered_count = int(filtered.get("num_matches", len(filtered.get("matches", []))))
        entry["filtered_match_count"] = filtered_count

        if filtered is not match_data and filtered_count >= 8:
            kpts1_filt = filtered["keypoints1"]
            kpts2_filt = filtered["keypoints2"]
        else:
            kpts1_filt = kpts1_raw
            kpts2_filt = kpts2_raw

        entry["filtered"] = {
            "colmap": evaluate_solver(
                "colmap",
                lambda: pose_estimator._estimate_pose_colmap(  # pylint: disable=protected-access
                    kpts1_filt, kpts2_filt, intrinsics, intrinsics
                ),
            ),
            "opencv": evaluate_solver(
                "opencv",
                lambda: pose_estimator._estimate_pose_opencv(  # pylint: disable=protected-access
                    kpts1_filt, kpts2_filt, intrinsics, intrinsics
                ),
            ),
        }

        diagnostics.append(entry)

    return diagnostics


def main():
    parser = argparse.ArgumentParser(description="Inspect per-pair pose estimation diagnostics.")
    parser.add_argument("--images", type=Path, required=True, help="Directory containing input images.")
    parser.add_argument("--ground_truth", type=Path, help="Path to synthetic ground truth JSON.")
    parser.add_argument("--config", type=Path, default=Path("config/base_config.yaml"), help="Hydra config file.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path for diagnostics.")
    parser.add_argument("--max_pairs", type=int, help="Optional limit on number of pairs to report.")
    args = parser.parse_args()

    if not args.images.exists():
        raise FileNotFoundError(f"Image directory not found: {args.images}")

    cfg = OmegaConf.load(args.config)
    cfg_obj = OmegaConf.to_object(cfg)
    config_ns = to_namespace(cfg_obj)

    feature_extractor = FeatureExtractor(config_ns)
    feature_matcher = FeatureMatcher(config_ns)
    pose_estimator = PoseEstimator(cfg_obj)

    image_paths = sorted(
        [p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
    )
    if len(image_paths) < 2:
        raise ValueError(f"Need at least two images in {args.images}; found {len(image_paths)}")

    print(f"[info] Loading {len(image_paths)} images from {args.images}")
    features = feature_extractor.extract_features_batch(image_paths)
    raw_matches = feature_matcher.match_all_pairs(features)

    intrinsics = np.array(
        [
            [800.0, 0.0, 320.0],
            [0.0, 800.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )

    cameras_gt = None
    if args.ground_truth:
        gt_data = load_ground_truth(args.ground_truth)
        cameras_gt = {name: camera for name, camera in gt_data.cameras.items()}

    diagnostics = gather_pose_diagnostics(
        features,
        raw_matches,
        pose_estimator,
        feature_matcher,
        intrinsics,
        cameras_gt,
    )

    if args.max_pairs:
        diagnostics = diagnostics[: args.max_pairs]

    print(json.dumps(diagnostics, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(diagnostics, indent=2))
        print(f"[info] Diagnostics written to {args.output}")


if __name__ == "__main__":
    main()
