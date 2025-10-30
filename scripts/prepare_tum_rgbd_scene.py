#!/usr/bin/env python3
"""
Prepare a TUM RGB-D sequence for the ModernSFM pipeline.

This script copies RGB frames, associates them with ground-truth poses, and
emits intrinsics/pose metadata in JSON format that our evaluation tools can read.
"""

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


DEFAULT_INTRINSICS = {
    "freiburg1": {"fx": 517.3, "fy": 516.5, "cx": 318.6, "cy": 255.3},
    "freiburg2": {"fx": 520.9, "fy": 521.0, "cx": 325.1, "cy": 249.7},
    "freiburg3": {"fx": 535.4, "fy": 539.2, "cx": 320.1, "cy": 247.6},
}


@dataclass
class PoseSample:
    timestamp: float
    position: np.ndarray  # shape (3,)
    rotation_matrix: np.ndarray  # shape (3, 3)


def load_rgb_list(rgb_txt: Path, max_frames: Optional[int] = None) -> List[Tuple[float, Path]]:
    entries: List[Tuple[float, Path]] = []
    with rgb_txt.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            timestamp = float(parts[0])
            rel_path = Path(parts[1])
            entries.append((timestamp, rel_path))
            if max_frames and len(entries) >= max_frames:
                break
    return entries


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion (qx, qy, qz, qw) to rotation matrix."""
    qx, qy, qz, qw = q
    norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm == 0:
        raise ValueError("Invalid quaternion with zero norm")
    qx, qy, qz, qw = qx / norm, qy / norm, qz / norm, qw / norm
    R = np.array(
        [
            [1 - 2 * (qy ** 2 + qz ** 2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx ** 2 + qz ** 2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx ** 2 + qy ** 2)],
        ],
        dtype=np.float64,
    )
    return R


def load_ground_truth(gt_txt: Path) -> List[PoseSample]:
    poses: List[PoseSample] = []
    with gt_txt.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                continue
            timestamp = float(parts[0])
            tx, ty, tz = map(float, parts[1:4])
            qx, qy, qz, qw = map(float, parts[4:8])
            R_wc = quaternion_to_rotation_matrix(np.array([qx, qy, qz, qw], dtype=np.float64))
            pose = PoseSample(timestamp=timestamp, position=np.array([tx, ty, tz], dtype=np.float64), rotation_matrix=R_wc)
            poses.append(pose)
    poses.sort(key=lambda p: p.timestamp)
    return poses


def associate_poses(
    rgb_list: List[Tuple[float, Path]],
    ground_truth: List[PoseSample],
    max_time_diff: float = 0.02,
) -> List[Tuple[float, Path, PoseSample]]:
    """Associate each RGB frame with the nearest ground-truth pose."""
    associated: List[Tuple[float, Path, PoseSample]] = []
    gt_idx = 0
    for timestamp, rel_path in rgb_list:
        while gt_idx + 1 < len(ground_truth) and abs(ground_truth[gt_idx + 1].timestamp - timestamp) < abs(ground_truth[gt_idx].timestamp - timestamp):
            gt_idx += 1
        pose = ground_truth[gt_idx]
        if abs(pose.timestamp - timestamp) <= max_time_diff:
            associated.append((timestamp, rel_path, pose))
        else:
            print(f"[warning] No ground-truth pose within {max_time_diff}s for frame at {timestamp:.6f}; skipping.")
    return associated


def read_intrinsics(rgb_txt: Path, dataset_root: Path) -> Dict[str, float]:
    fx = fy = cx = cy = None
    with rgb_txt.open("r") as fh:
        for line in fh:
            if line.startswith("#"):
                if "fx" in line and "fy" in line and "cx" in line and "cy" in line:
                    parts = line.replace(",", " ").replace("=", " ").split()
                    values: Dict[str, float] = {}
                    for idx, token in enumerate(parts):
                        if token in {"fx", "fy", "cx", "cy"} and idx + 1 < len(parts):
                            try:
                                values[token] = float(parts[idx + 1])
                            except ValueError:
                                pass
                    fx = values.get("fx", fx)
                    fy = values.get("fy", fy)
                    cx = values.get("cx", cx)
                    cy = values.get("cy", cy)
    if None in (fx, fy, cx, cy):
        root_name = dataset_root.name.lower()
        fallback: Optional[Dict[str, float]] = None
        for key, values in DEFAULT_INTRINSICS.items():
            if key in root_name:
                fallback = values
                break
        if fallback is None:
            raise RuntimeError("Could not parse intrinsics from rgb.txt comments, and no fallback found for dataset.")
        print(f"[warning] Using fallback intrinsics for {root_name}: {fallback}")
        return fallback
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a TUM RGB-D sequence for ModernSFM.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path to the extracted TUM RGB-D sequence (e.g., rgbd_dataset_freiburg1_xyz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tum_rgbd_prepared/fr1_xyz"),
        help="Directory to write processed data.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Optional maximum number of frames to process (useful for quick tests).",
    )
    parser.add_argument(
        "--copy-mode",
        choices=["copy", "symlink"],
        default="copy",
        help="Whether to copy or symlink images into the output folder.",
    )
    parser.add_argument(
        "--time-diff",
        type=float,
        default=0.02,
        help="Maximum allowed time difference (seconds) between RGB frame and ground-truth pose.",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root
    rgb_txt = dataset_root / "rgb.txt"
    gt_txt = dataset_root / "groundtruth.txt"
    rgb_dir = dataset_root / "rgb"

    if not rgb_txt.exists() or not gt_txt.exists() or not rgb_dir.exists():
        raise FileNotFoundError("Expected rgb.txt, groundtruth.txt, and rgb/ directory under dataset root.")

    rgb_list = load_rgb_list(rgb_txt, max_frames=args.max_frames)
    ground_truth = load_ground_truth(gt_txt)
    associations = associate_poses(rgb_list, ground_truth, max_time_diff=args.time_diff)

    if not associations:
        raise RuntimeError("No frames had associated ground-truth poses.")

    intrinsics = read_intrinsics(rgb_txt, dataset_root)

    images_output = args.output / "images"
    data_output = args.output / "data"
    images_output.mkdir(parents=True, exist_ok=True)
    data_output.mkdir(parents=True, exist_ok=True)

    frame_entries = []
    for idx, (timestamp, rel_path, pose_sample) in enumerate(associations):
        src_path = dataset_root / rel_path
        if not src_path.exists():
            print(f"[warning] Missing image file {src_path}; skipping.")
            continue
        filename = f"frame_{idx:05d}{src_path.suffix.lower()}"
        dst_path = images_output / filename
        if args.copy_mode == "symlink":
            if dst_path.exists():
                dst_path.unlink()
            dst_path.symlink_to(src_path.resolve())
        else:
            shutil.copy2(src_path, dst_path)

        R_wc = pose_sample.rotation_matrix
        t_wc = pose_sample.position.reshape(3, 1)
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc

        frame_entries.append({
            "frame_id": idx,
            "timestamp": timestamp,
            "image_path": str(Path("images") / filename),
            "rotation_matrix": R_cw.tolist(),
            "translation": t_cw.flatten().tolist(),
        })

    if not frame_entries:
        raise RuntimeError("No frame entries were written.")

    first_image = images_output / Path(frame_entries[0]["image_path"]).name
    with Image.open(first_image) as img:
        width, height = img.size

    intrinsics_json = {
        "intrinsics": [
            [intrinsics["fx"], 0.0, intrinsics["cx"]],
            [0.0, intrinsics["fy"], intrinsics["cy"]],
            [0.0, 0.0, 1.0],
        ],
        "image_size": [width, height],
        "sequence": dataset_root.name,
    }
    (data_output / "camera_intrinsics.json").write_text(json.dumps(intrinsics_json, indent=2))
    (data_output / "ground_truth_poses.json").write_text(json.dumps({"frames": frame_entries}, indent=2))

    print(f"[info] Prepared {len(frame_entries)} frames at {args.output}")


if __name__ == "__main__":
    main()
