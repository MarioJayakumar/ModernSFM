#!/usr/bin/env python3
"""
Evaluate a ModernSFM reconstruction against TUM RGB-D ground truth.

Usage:
    python -m scripts.evaluate_tum_reconstruction \\
        --ground-truth data/tum_rgbd/fr1_xyz/data/ground_truth_poses.json \\
        --reconstruction outputs/reconstruction_tum_fr1_10/reconstruction_2025-10-27_22-45-01
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_ground_truth(path: Path) -> Dict[int, Dict[str, float]]:
    data = json.loads(path.read_text())
    frames = data.get("frames", [])
    gt = {}
    for entry in frames:
        frame_id = int(entry["frame_id"])
        gt[frame_id] = {
            "timestamp": entry["timestamp"],
            "rotation_matrix": np.array(entry["rotation_matrix"], dtype=np.float64),
            "translation": np.array(entry["translation"], dtype=np.float64),
        }
    return gt


def load_reconstruction(path: Path) -> Dict[int, Dict[str, float]]:
    poses_path = path / "data" / "camera_poses.json"
    if not poses_path.exists():
        raise FileNotFoundError(f"camera_poses.json not found in {path}")
    data = json.loads(poses_path.read_text())
    recon = {}
    for entry in data:
        frame_id = int(entry["camera_id"])
        recon[frame_id] = {
            "rotation_matrix": np.array(entry["R"], dtype=np.float64),
            "translation": np.array(entry["t"], dtype=np.float64),
        }
    return recon


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    R_rel = R_gt @ R_est.T
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_error(R_gt: np.ndarray, t_gt: np.ndarray, R_est: np.ndarray, t_est: np.ndarray) -> float:
    # Both t vectors are camera-to-world; convert to camera centers for comparison
    C_gt = -R_gt.T @ t_gt.reshape(3, 1)
    C_est = -R_est.T @ t_est.reshape(3, 1)
    return float(np.linalg.norm(C_gt - C_est))


def evaluate(gt_data: Dict[int, Dict[str, float]], recon_data: Dict[int, Dict[str, float]]) -> Tuple[List[float], List[float], List[int]]:
    rotation_errors = []
    translation_errors = []
    evaluated_frames = []

    for frame_id, gt_pose in gt_data.items():
        if frame_id not in recon_data:
            continue
        R_gt = gt_pose["rotation_matrix"]
        t_gt = gt_pose["translation"]
        R_est = recon_data[frame_id]["rotation_matrix"]
        t_est = recon_data[frame_id]["translation"]

        rot_err = rotation_error_deg(R_gt, R_est)
        trans_err = translation_error(R_gt, t_gt, R_est, t_est)

        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)
        evaluated_frames.append(frame_id)

    return rotation_errors, translation_errors, evaluated_frames


def main():
    parser = argparse.ArgumentParser(description="Evaluate reconstruction against TUM RGB-D ground truth.")
    parser.add_argument("--ground-truth", type=Path, required=True, help="Path to ground_truth_poses.json")
    parser.add_argument("--reconstruction", type=Path, required=True, help="Path to reconstruction output directory")
    parser.add_argument("--max-frames", type=int, help="Optional limit on number of frames to evaluate")
    args = parser.parse_args()

    gt = load_ground_truth(args.ground_truth)
    recon = load_reconstruction(args.reconstruction)

    if args.max_frames is not None:
        gt = {fid: pose for fid, pose in gt.items() if fid < args.max_frames}

    rotation_errors, translation_errors, frames = evaluate(gt, recon)

    if not frames:
        print("No overlapping frames between ground truth and reconstruction.")
        return

    print(f"Evaluated frames: {len(frames)}")
    print(f"Rotation error (deg): mean={np.mean(rotation_errors):.2f}, median={np.median(rotation_errors):.2f}, max={np.max(rotation_errors):.2f}")
    print(f"Translation error (m): mean={np.mean(translation_errors):.3f}, median={np.median(translation_errors):.3f}, max={np.max(translation_errors):.3f}")


if __name__ == "__main__":
    main()
