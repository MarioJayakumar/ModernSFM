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


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute camera center in world coordinates from extrinsics."""
    return -R.T @ t.reshape(3, 1)


def rotation_error_deg(R_gt: np.ndarray, R_est: np.ndarray) -> float:
    R_rel = R_gt @ R_est.T
    trace = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def umeyama_alignment(centers_est: np.ndarray, centers_gt: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate similarity transform (scale * R * x + t) that aligns estimated centers to ground truth."""
    n = centers_est.shape[0]
    if n == 0:
        raise ValueError("No camera centers provided for alignment.")

    centers_est = centers_est.astype(np.float64)
    centers_gt = centers_gt.astype(np.float64)

    if n == 1:
        # With a single camera the rotation is ambiguous; treat as pure translation.
        return 1.0, np.eye(3), centers_gt[0] - centers_est[0]

    mu_est = centers_est.mean(axis=0)
    mu_gt = centers_gt.mean(axis=0)

    X = centers_est - mu_est
    Y = centers_gt - mu_gt

    cov = (Y.T @ X) / n
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    var = np.sum(X**2) / n
    scale = S.sum() / var if var > 1e-9 else 1.0
    t = mu_gt - scale * R @ mu_est

    return float(scale), R, t


def evaluate(gt_data: Dict[int, Dict[str, float]], recon_data: Dict[int, Dict[str, float]]) -> Tuple[List[float], List[float], List[int], Dict[str, float]]:
    rotation_errors: List[float] = []
    translation_errors: List[float] = []
    evaluated_frames: List[int] = []

    centers_gt: List[np.ndarray] = []
    centers_est: List[np.ndarray] = []
    rotations_gt: List[np.ndarray] = []
    rotations_est: List[np.ndarray] = []

    for frame_id, gt_pose in gt_data.items():
        if frame_id not in recon_data:
            continue

        R_gt = gt_pose["rotation_matrix"]
        t_gt = gt_pose["translation"]
        R_est = recon_data[frame_id]["rotation_matrix"]
        t_est = recon_data[frame_id]["translation"]

        centers_gt.append(camera_center(R_gt, t_gt).reshape(3))
        centers_est.append(camera_center(R_est, t_est).reshape(3))
        rotations_gt.append(R_gt)
        rotations_est.append(R_est)
        evaluated_frames.append(frame_id)

    if not evaluated_frames:
        return rotation_errors, translation_errors, evaluated_frames, {}

    centers_gt_arr = np.vstack(centers_gt)
    centers_est_arr = np.vstack(centers_est)

    # Orientation alignment: find global rotation Q such that R_est @ Q ≈ R_gt
    orientation_stack = np.zeros((3, 3), dtype=np.float64)
    for R_gt, R_est in zip(rotations_gt, rotations_est):
        orientation_stack += R_est.T @ R_gt
    U, _, Vt = np.linalg.svd(orientation_stack)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        U[:, -1] *= -1
        Q = U @ Vt

    # Rotate camera centers accordingly (C transforms with Q^T)
    centers_est_oriented = (Q.T @ centers_est_arr.T).T

    # Solve for scale and translation with orientation fixed
    mu_est = centers_est_oriented.mean(axis=0)
    mu_gt = centers_gt_arr.mean(axis=0)
    X = centers_est_oriented - mu_est
    Y = centers_gt_arr - mu_gt
    denom = np.sum(X * X)
    scale = float(np.sum(X * Y) / denom) if denom > 1e-9 else 1.0
    translation = mu_gt - scale * mu_est

    alignment_info = {
        "scale": scale,
        "orientation_correction": Q.tolist(),
        "translation": translation.tolist(),
    }

    for idx, frame_id in enumerate(evaluated_frames):
        R_gt = rotations_gt[idx]
        R_est = rotations_est[idx]

        C_oriented = centers_est_oriented[idx]
        C_aligned = scale * C_oriented + translation
        R_aligned = R_est @ Q

        rot_err = rotation_error_deg(R_gt, R_aligned)
        trans_err = np.linalg.norm(centers_gt_arr[idx] - C_aligned)

        rotation_errors.append(rot_err)
        translation_errors.append(trans_err)

    return rotation_errors, translation_errors, evaluated_frames, alignment_info


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

    rotation_errors, translation_errors, frames, alignment = evaluate(gt, recon)

    if not frames:
        print("No overlapping frames between ground truth and reconstruction.")
        return

    print(f"Evaluated frames: {len(frames)}")
    print(f"Rotation error (deg): mean={np.mean(rotation_errors):.2f}, median={np.median(rotation_errors):.2f}, max={np.max(rotation_errors):.2f}")
    print(f"Translation error (m): mean={np.mean(translation_errors):.3f}, median={np.median(translation_errors):.3f}, max={np.max(translation_errors):.3f}")
    if alignment:
        print("Applied alignment:")
        print(f"  Scale: {alignment['scale']:.6f}")
        print(f"  Orientation correction (right-multiplied):\n{np.array(alignment['orientation_correction'])}")
        print(f"  Translation: {np.array(alignment['translation'])}")


if __name__ == "__main__":
    main()
