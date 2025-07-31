"""
Simple script to visualize 3D point cloud data from various file formats.

This script loads existing point cloud data and visualizes it using the
SfMVisualizer, perfect for viewing results over SSH.

Supported formats:
- PLY files (standard point cloud format)
- NPY files (numpy arrays)
- TXT files (space-separated x y z [r g b])
- JSON files (with points_3d array)

Usage:
    python scripts/view_results.py path/to/pointcloud.ply
    python scripts/view_results.py path/to/points.npy --method export
    python scripts/view_results.py path/to/data.txt --method terminal
"""

import sys
from pathlib import Path
import logging
import argparse
import numpy as np
import json

# Add the project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from omegaconf import OmegaConf
from main.utils.visualization import create_visualizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_ply_file(filepath):
    """Load point cloud from PLY file."""
    logger.info(f"Loading PLY file: {filepath}")
    
    points = []
    colors = []
    
    with open(filepath, 'r') as f:
        # Read header
        line = f.readline().strip()
        if line != 'ply':
            raise ValueError("Not a valid PLY file")
        
        vertex_count = 0
        has_colors = False
        
        # Parse header
        while True:
            line = f.readline().strip()
            if line == 'end_header':
                break
            elif line.startswith('element vertex'):
                vertex_count = int(line.split()[-1])
            elif 'property uchar red' in line or 'property uchar green' in line:
                has_colors = True
        
        # Read vertex data
        for _ in range(vertex_count):
            line = f.readline().strip()
            parts = line.split()
            
            # Extract coordinates
            x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            points.append([x, y, z])
            
            # Extract colors if available
            if has_colors and len(parts) >= 6:
                r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                colors.append([r/255.0, g/255.0, b/255.0])
    
    points = np.array(points)
    colors = np.array(colors) if colors else None
    
    logger.info(f"Loaded {len(points)} points from PLY file")
    if colors is not None:
        logger.info("Colors found in PLY file")
    
    return points, colors


def load_npy_file(filepath):
    """Load point cloud from numpy file."""
    logger.info(f"Loading NPY file: {filepath}")
    
    data = np.load(filepath)
    
    if data.ndim == 2 and data.shape[1] >= 3:
        points = data[:, :3]
        colors = data[:, 3:6] if data.shape[1] >= 6 else None
        
        logger.info(f"Loaded {len(points)} points from NPY file")
        if colors is not None:
            logger.info("Colors found in NPY file")
        
        return points, colors
    else:
        raise ValueError(f"Invalid NPY file shape: {data.shape}. Expected (N, 3) or (N, 6)")


def load_txt_file(filepath):
    """Load point cloud from text file (x y z [r g b])."""
    logger.info(f"Loading TXT file: {filepath}")
    
    data = np.loadtxt(filepath)
    
    if data.ndim == 2 and data.shape[1] >= 3:
        points = data[:, :3]
        colors = data[:, 3:6] if data.shape[1] >= 6 else None
        
        # If colors are in 0-255 range, normalize to 0-1
        if colors is not None and np.max(colors) > 1.0:
            colors = colors / 255.0
        
        logger.info(f"Loaded {len(points)} points from TXT file")
        if colors is not None:
            logger.info("Colors found in TXT file")
        
        return points, colors
    else:
        raise ValueError(f"Invalid TXT file shape: {data.shape}. Expected (N, 3) or (N, 6)")


def load_json_file(filepath):
    """Load point cloud from JSON file."""
    logger.info(f"Loading JSON file: {filepath}")
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'points_3d' in data:
        points = np.array(data['points_3d'])
        colors = np.array(data['colors']) if 'colors' in data else None
        
        logger.info(f"Loaded {len(points)} points from JSON file")
        if colors is not None:
            logger.info("Colors found in JSON file")
        
        return points, colors
    else:
        raise ValueError("JSON file must contain 'points_3d' array")


def load_point_cloud(filepath):
    """Load point cloud from various file formats."""
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    extension = filepath.suffix.lower()
    
    if extension == '.ply':
        return load_ply_file(filepath)
    elif extension == '.npy':
        return load_npy_file(filepath)
    elif extension in ['.txt', '.xyz']:
        return load_txt_file(filepath)
    elif extension == '.json':
        return load_json_file(filepath)
    else:
        raise ValueError(f"Unsupported file format: {extension}")


def create_sample_data():
    """Create sample point cloud data for demonstration."""
    logger.info("Creating sample point cloud data...")
    
    np.random.seed(42)
    
    # Create a simple 3D scene
    n_points = 200
    
    # Ground plane
    ground_x = np.random.uniform(-10, 10, n_points//2)
    ground_y = np.random.uniform(-10, 10, n_points//2)
    ground_z = np.random.normal(0, 0.5, n_points//2)
    
    # Elevated structure
    struct_x = np.random.uniform(-3, 3, n_points//2)
    struct_y = np.random.uniform(-3, 3, n_points//2)
    struct_z = np.random.uniform(2, 8, n_points//2)
    
    # Combine points
    points = np.column_stack([
        np.concatenate([ground_x, struct_x]),
        np.concatenate([ground_y, struct_y]),
        np.concatenate([ground_z, struct_z])
    ])
    
    # Create colors based on height
    z_normalized = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]))
    colors = np.zeros((len(points), 3))
    colors[:, 0] = z_normalized  # Red increases with height
    colors[:, 1] = 1 - z_normalized  # Green decreases with height
    colors[:, 2] = 0.3  # Constant blue
    
    logger.info(f"Created sample data with {len(points)} points")
    
    return points, colors


def main():
    parser = argparse.ArgumentParser(description='Visualize 3D point cloud data')
    parser.add_argument('file', nargs='?', help='Path to point cloud file (PLY, NPY, TXT, JSON)')
    parser.add_argument('--method', choices=['auto', 'matplotlib', 'export', 'web', 'terminal'], 
                       default='auto', help='Visualization method')
    parser.add_argument('--title', default='Point Cloud Viewer', help='Title for visualization')
    parser.add_argument('--sample', action='store_true', help='Use sample data instead of file')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("3D POINT CLOUD VIEWER")
    logger.info("=" * 60)
    
    try:
        # Load configuration
        config_path = Path(__file__).parent.parent / "config" / "base_config.yaml"
        with open(config_path, 'r') as f:
            cfg = OmegaConf.load(f)
        
        # Create visualizer
        visualizer = create_visualizer(dict(cfg))
        
        # Load data
        if args.sample:
            points, colors = create_sample_data()
            title = f"{args.title} - Sample Data"
        elif args.file:
            points, colors = load_point_cloud(args.file)
            title = f"{args.title} - {Path(args.file).name}"
        else:
            logger.error("Please provide a file path or use --sample")
            logger.info("\nUsage examples:")
            logger.info("  python scripts/view_results.py outputs/visualizations/points.ply")
            logger.info("  python scripts/view_results.py data.npy --method export")
            logger.info("  python scripts/view_results.py --sample --method terminal")
            return
        
        # Visualize
        logger.info(f"Visualizing with method: {args.method}")
        result_path = visualizer.visualize_point_cloud(
            points, colors, title=title, method=args.method
        )
        
        if result_path:
            logger.info(f"✓ Visualization completed: {result_path}")
        else:
            logger.warning("Visualization completed but no output path returned")
        
        # Print usage instructions
        logger.info("\n" + "=" * 60)
        logger.info("USAGE EXAMPLES")
        logger.info("=" * 60)
        logger.info("# View PLY file (auto-detect best method)")
        logger.info("python scripts/view_results.py outputs/visualizations/points.ply")
        logger.info("")
        logger.info("# View numpy array with export method (creates downloadable files)")
        logger.info("python scripts/view_results.py data/points.npy --method export")
        logger.info("")
        logger.info("# Quick terminal preview")
        logger.info("python scripts/view_results.py data.txt --method terminal")
        logger.info("")
        logger.info("# View sample data")
        logger.info("python scripts/view_results.py --sample")
        logger.info("")
        logger.info("# Interactive plot (if you have X11 forwarding)")
        logger.info("python scripts/view_results.py points.ply --method matplotlib")
        
        logger.info("\n" + "=" * 60)
        logger.info("SUPPORTED FILE FORMATS")
        logger.info("=" * 60)
        logger.info("• PLY files: Standard point cloud format")
        logger.info("• NPY files: Numpy arrays (N,3) or (N,6) with colors")
        logger.info("• TXT files: Space-separated x y z [r g b]")
        logger.info("• JSON files: {'points_3d': [[x,y,z], ...], 'colors': [[r,g,b], ...]}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
