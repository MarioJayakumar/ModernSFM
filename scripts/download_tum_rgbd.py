#!/usr/bin/env python3
"""
Download a TUM RGB-D sequence (RGB images, depth frames, and ground-truth trajectory).
"""

import argparse
import hashlib
import tarfile
from pathlib import Path
from typing import Dict, Tuple

import requests
from tqdm import tqdm


TUM_SEQUENCES: Dict[str, Tuple[str, str]] = {
    "fr1_xyz": (
        "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_xyz.tgz",
        "9a0e5aaadc8f4b6e8d0d3bc2c4d0f0af",
    ),
    "fr1_plant": (
        "https://vision.in.tum.de/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_plant.tgz",
        "7f5d29a0b2b7cf3a06f8f287a3b883d0",
    ),
    "fr3_office": (
        "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_office.tgz",
        "1e0a73d8da5b6a1cb43cad9dae1ab08d",
    ),
}


def download_file(url: str, dest: Path) -> None:
    """Stream a file from a URL with progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {dest.name}") as pbar:
        with dest.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    fh.write(chunk)
                    pbar.update(len(chunk))


def verify_md5(filepath: Path, expected_md5: str) -> str:
    """Verify file checksum."""
    hash_md5 = hashlib.md5()
    with filepath.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def extract_tgz(tar_path: Path, target_dir: Path) -> None:
    """Extract .tgz archive."""
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=target_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a TUM RGB-D dataset sequence.")
    parser.add_argument(
        "--sequence",
        type=str,
        choices=TUM_SEQUENCES.keys(),
        required=True,
        help="Sequence ID (e.g., fr1_xyz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/tum_rgbd"),
        help="Directory to store the downloaded sequence.",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded .tgz archive after extraction.",
    )
    parser.add_argument(
        "--skip-md5",
        action="store_true",
        help="Skip MD5 verification (use only if you trust the download source).",
    )
    args = parser.parse_args()

    url, md5 = TUM_SEQUENCES[args.sequence]
    output_dir = args.output
    archive_path = output_dir / f"{args.sequence}.tgz"

    output_dir.mkdir(parents=True, exist_ok=True)

    if not archive_path.exists():
        download_file(url, archive_path)
    else:
        print(f"[info] Archive already exists at {archive_path}")

    if not args.skip_md5:
        print("[info] Verifying checksum...")
        computed_md5 = verify_md5(archive_path, md5)
        if computed_md5 != md5:
            print(f"[error] Expected MD5: {md5}")
            print(f"[error] Computed MD5: {computed_md5}")
            raise RuntimeError("MD5 checksum mismatch. Delete the archive and retry, or rerun with --skip-md5.")
        print("[info] Checksum verified.")
    else:
        print("[warning] Skipping MD5 verification as requested.")

    print("[info] Extracting archive...")
    extract_tgz(archive_path, output_dir)
    print(f"[info] Sequence extracted to {output_dir / args.sequence}")

    if not args.keep_archive:
        archive_path.unlink()
        print("[info] Removed downloaded archive.")


if __name__ == "__main__":
    main()
