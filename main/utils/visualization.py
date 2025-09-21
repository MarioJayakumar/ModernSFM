"""
3D Visualization utilities for Modern SfM Pipeline

This module provides multiple visualization options that work over SSH connections:
1. X11 forwarding with matplotlib (interactive 3D)
2. Export to standard formats (PLY, OBJ) for local viewing
3. SSH tunnel web server (optional)
4. Terminal-based visualization for quick previews

Author: ModernSFM Pipeline
Date: July 28th, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import os
import subprocess
import tempfile
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver


class SfMVisualizer:
    """
    Multi-modal 3D visualization for SfM results that works over SSH.
    
    Provides several visualization options:
    - Interactive matplotlib 3D plots (with X11 forwarding)
    - Export to standard 3D formats (PLY, OBJ)
    - Optional web-based viewer with SSH tunneling
    - Terminal-based ASCII visualization for quick previews
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.viz_config = config.get('visualization', {})
        
        # Visualization settings
        self.point_size = self.viz_config.get('point_size', 2.0)
        self.camera_size = self.viz_config.get('camera_size', 0.1)
        self.save_intermediate = self.viz_config.get('save_intermediate', True)
        
        # Output settings
        self.output_dir = Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.info("Initialized SfMVisualizer with multiple SSH-compatible options")
    
    def visualize_point_cloud(self,
                            points_3d: np.ndarray,
                            colors: Optional[np.ndarray] = None,
                            camera_poses: Optional[Dict[int, np.ndarray]] = None,
                            title: str = "3D Point Cloud",
                            method: str = "auto") -> str:
        """
        Visualize 3D point cloud using the best available method.
        
        Args:
            points_3d: [N, 3] array of 3D points
            colors: [N, 3] array of RGB colors (0-1 range), optional
            camera_poses: Dict mapping camera_id -> [3, 4] pose matrix, optional
            title: Title for the visualization
            method: 'auto', 'matplotlib', 'export', 'web', 'terminal'
            
        Returns:
            Path to saved visualization or URL if web-based
        """
        if len(points_3d) == 0:
            logging.warning("No points to visualize")
            return ""
        
        # Auto-detect best method
        if method == "auto":
            method = self._detect_best_method()
        
        logging.info(f"Using visualization method: {method}")
        
        if method == "matplotlib":
            return self._visualize_matplotlib(points_3d, colors, camera_poses, title)
        elif method == "export":
            return self._export_formats(points_3d, colors, camera_poses, title)
        elif method == "web":
            return self._visualize_web(points_3d, colors, camera_poses, title)
        elif method == "terminal":
            return self._visualize_terminal(points_3d, colors, camera_poses, title)
        else:
            logging.error(f"Unknown visualization method: {method}")
            return ""
    
    def visualize_reconstruction(self,
                               triangulation_result: Dict,
                               camera_poses: Optional[Dict[int, np.ndarray]] = None,
                               title: str = "SfM Reconstruction",
                               method: str = "auto") -> str:
        """
        Visualize complete reconstruction results.
        
        Args:
            triangulation_result: Result dict from Triangulator
            camera_poses: Camera poses for visualization
            title: Title for visualization
            method: Visualization method to use
            
        Returns:
            Path to visualization or URL
        """
        points_3d = triangulation_result.get('points_3d')
        valid_mask = triangulation_result.get('valid_mask')
        
        if points_3d is None:
            logging.warning("No triangulation results to visualize")
            return ""
        
        # Filter to valid points only
        if valid_mask is not None:
            points_3d = points_3d[valid_mask]
        
        # Color points by quality metrics
        colors = None
        if 'reprojection_errors' in triangulation_result:
            errors = triangulation_result['reprojection_errors']
            if valid_mask is not None:
                errors = errors[valid_mask]
            colors = self._error_to_color(errors)
        
        return self.visualize_point_cloud(points_3d, colors, camera_poses, title, method)
    
    def _detect_best_method(self) -> str:
        """Auto-detect the best visualization method for current environment."""
        
        # Check if we're in SSH session
        if self._is_ssh_session():
            # Check if X11 forwarding is available
            if self._has_x11_forwarding():
                logging.info("SSH with X11 forwarding detected - using matplotlib")
                return "matplotlib"
            else:
                logging.info("SSH without X11 forwarding - using export method")
                return "export"
        else:
            # Local session - use matplotlib
            return "matplotlib"
    
    def _is_ssh_session(self) -> bool:
        """Check if we're running in an SSH session."""
        return (os.environ.get('SSH_CLIENT') is not None or 
                os.environ.get('SSH_TTY') is not None or
                os.environ.get('SSH_CONNECTION') is not None)
    
    def _has_x11_forwarding(self) -> bool:
        """Check if X11 forwarding is available."""
        return os.environ.get('DISPLAY') is not None
    
    def _visualize_matplotlib(self,
                            points_3d: np.ndarray,
                            colors: Optional[np.ndarray],
                            camera_poses: Optional[Dict[int, np.ndarray]],
                            title: str) -> str:
        """Create interactive matplotlib 3D visualization."""
        
        try:
            # Set matplotlib backend for SSH
            if self._is_ssh_session():
                plt.switch_backend('TkAgg')  # Good for X11 forwarding
            
            fig = plt.figure(figsize=(15, 10))
            
            # Main 3D plot
            ax_3d = fig.add_subplot(221, projection='3d')
            
            # Plot points
            if colors is not None:
                ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                            c=colors, s=self.point_size**2, alpha=0.6)
            else:
                ax_3d.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                            c='blue', s=self.point_size**2, alpha=0.6)
            
            # Plot cameras if available
            if camera_poses:
                self._plot_cameras_matplotlib(ax_3d, camera_poses)
            
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title(title)
            
            # Add coordinate projections
            ax_xy = fig.add_subplot(222)
            ax_xy.scatter(points_3d[:, 0], points_3d[:, 1], 
                         c=colors[:, 0] if colors is not None else 'blue', 
                         s=1, alpha=0.6)
            ax_xy.set_xlabel('X')
            ax_xy.set_ylabel('Y')
            ax_xy.set_title('XY Projection')
            ax_xy.axis('equal')
            
            ax_xz = fig.add_subplot(223)
            ax_xz.scatter(points_3d[:, 0], points_3d[:, 2], 
                         c=colors[:, 1] if colors is not None else 'blue', 
                         s=1, alpha=0.6)
            ax_xz.set_xlabel('X')
            ax_xz.set_ylabel('Z')
            ax_xz.set_title('XZ Projection')
            ax_xz.axis('equal')
            
            ax_yz = fig.add_subplot(224)
            ax_yz.scatter(points_3d[:, 1], points_3d[:, 2], 
                         c=colors[:, 2] if colors is not None else 'blue', 
                         s=1, alpha=0.6)
            ax_yz.set_xlabel('Y')
            ax_yz.set_ylabel('Z')
            ax_yz.set_title('YZ Projection')
            ax_yz.axis('equal')
            
            plt.tight_layout()
            
            # Save static version
            output_path = self.output_dir / f"{title.lower().replace(' ', '_')}_matplotlib.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logging.info(f"Saved static visualization to {output_path}")
            
            # Show interactive plot (works with X11 forwarding)
            if self._has_x11_forwarding():
                plt.show(block=False)
                logging.info("Interactive 3D plot opened (use mouse to rotate/zoom)")
                logging.info("Close the plot window to continue...")
            else:
                logging.info("X11 forwarding not available - saved static image only")
                plt.close()
            
            return str(output_path)
            
        except Exception as e:
            logging.error(f"Matplotlib visualization failed: {e}")
            return self._export_formats(points_3d, colors, camera_poses, title)
    
    def _plot_cameras_matplotlib(self, ax, camera_poses: Dict[int, np.ndarray]):
        """Add camera visualizations to matplotlib 3D plot."""
        
        for cam_id, pose in camera_poses.items():
            if pose.shape == (3, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
                
                # Camera center
                center = -R.T @ t
                
                # Camera coordinate axes
                scale = self.camera_size * 5
                axes = R.T * scale
                
                # Plot camera center
                ax.scatter([center[0]], [center[1]], [center[2]], 
                          c='red', s=50, marker='s')
                
                # Plot camera axes
                for i, color in enumerate(['red', 'green', 'blue']):
                    axis_end = center + axes[:, i]
                    ax.plot([center[0], axis_end[0]], 
                           [center[1], axis_end[1]], 
                           [center[2], axis_end[2]], 
                           color=color, linewidth=2)
                
                # Add camera ID label
                ax.text(center[0], center[1], center[2], f'  Cam{cam_id}', 
                       fontsize=8)
    
    def _export_formats(self,
                       points_3d: np.ndarray,
                       colors: Optional[np.ndarray],
                       camera_poses: Optional[Dict[int, np.ndarray]],
                       title: str) -> List[str]:
        """Export point cloud to standard 3D formats."""
        
        base_name = title.lower().replace(' ', '_')
        
        # Export PLY format (most common for point clouds)
        ply_path = self._export_ply(points_3d, colors, f"{base_name}.ply")
        
        # Export OBJ format (good for meshes later)
        obj_path = self._export_obj(points_3d, colors, f"{base_name}.obj")
        
        # Export camera poses separately
        if camera_poses:
            poses_path = self._export_camera_poses(camera_poses, f"{base_name}_cameras.json")
        
        # Create a simple HTML viewer file
        html_path = self._create_simple_html_viewer(points_3d, colors, f"{base_name}_viewer.html")
        
        logging.info(f"Exported point cloud to multiple formats:")
        logging.info(f"  PLY: {ply_path}")
        logging.info(f"  OBJ: {obj_path}")
        if camera_poses:
            logging.info(f"  Cameras: {poses_path}")
        logging.info(f"  HTML Viewer: {html_path}")
        
        # Provide viewing instructions
        self._print_viewing_instructions(ply_path, obj_path, html_path)
        
        return [str(ply_path), str(obj_path), str(html_path)]
    
    def _export_ply(self, points_3d: np.ndarray, colors: Optional[np.ndarray], filename: str) -> Path:
        """Export point cloud to PLY format."""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            # PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points_3d)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            
            if colors is not None:
                f.write("property uchar red\n")
                f.write("property uchar green\n")
                f.write("property uchar blue\n")
            
            f.write("end_header\n")
            
            # Point data
            for i, point in enumerate(points_3d):
                if colors is not None:
                    # Convert colors from [0,1] to [0,255]
                    r, g, b = (colors[i] * 255).astype(int)
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n")
                else:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
        
        return output_path
    
    def _export_obj(self, points_3d: np.ndarray, colors: Optional[np.ndarray], filename: str) -> Path:
        """Export point cloud to OBJ format."""
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w') as f:
            f.write("# OBJ file generated by ModernSFM\n")
            f.write(f"# {len(points_3d)} vertices\n")
            
            # Vertices
            for point in points_3d:
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # Colors as comments (OBJ doesn't have standard color support)
            if colors is not None:
                f.write("\n# Vertex colors (r g b)\n")
                for i, color in enumerate(colors):
                    f.write(f"# vc {i+1} {color[0]:.3f} {color[1]:.3f} {color[2]:.3f}\n")
        
        return output_path
    
    def _export_camera_poses(self, camera_poses: Dict[int, np.ndarray], filename: str) -> Path:
        """Export camera poses to JSON format."""
        
        output_path = self.output_dir / filename
        
        poses_data = {}
        for cam_id, pose in camera_poses.items():
            if pose.shape == (3, 4):
                R = pose[:3, :3]
                t = pose[:3, 3]
                center = (-R.T @ t).tolist()
                
                poses_data[str(cam_id)] = {
                    'rotation_matrix': R.tolist(),
                    'translation': t.tolist(),
                    'camera_center': center
                }
        
        with open(output_path, 'w') as f:
            json.dump(poses_data, f, indent=2)
        
        return output_path
    
    def _create_simple_html_viewer(self, points_3d: np.ndarray, colors: Optional[np.ndarray], filename: str) -> Path:
        """Create a simple HTML file that can be downloaded and viewed locally."""
        
        output_path = self.output_dir / filename
        
        # Sample points for performance (if too many)
        if len(points_3d) > 10000:
            indices = np.random.choice(len(points_3d), 10000, replace=False)
            points_sample = points_3d[indices]
            colors_sample = colors[indices] if colors is not None else None
        else:
            points_sample = points_3d
            colors_sample = colors
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>SfM Point Cloud Viewer</title>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #info {{ position: absolute; top: 10px; left: 10px; color: white; background: rgba(0,0,0,0.7); padding: 10px; }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="info">
        <h3>SfM Point Cloud</h3>
        <p>Points: {len(points_sample)}</p>
        <p>Left click: rotate | Right click: pan | Scroll: zoom</p>
    </div>
    
    <script>
        // Point cloud data
        const points = {json.dumps(points_sample.tolist())};
        const colors = {json.dumps(colors_sample.tolist()) if colors_sample is not None else 'null'};
        
        // Scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.body.appendChild(renderer.domElement);
        
        // Create point cloud
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(points.length * 3);
        
        for (let i = 0; i < points.length; i++) {{
            positions[i * 3] = points[i][0];
            positions[i * 3 + 1] = points[i][1];
            positions[i * 3 + 2] = points[i][2];
        }}
        
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        
        if (colors) {{
            const colorArray = new Float32Array(colors.length * 3);
            for (let i = 0; i < colors.length; i++) {{
                colorArray[i * 3] = colors[i][0];
                colorArray[i * 3 + 1] = colors[i][1];
                colorArray[i * 3 + 2] = colors[i][2];
            }}
            geometry.setAttribute('color', new THREE.BufferAttribute(colorArray, 3));
        }}
        
        const material = new THREE.PointsMaterial({{
            size: 2,
            vertexColors: colors ? true : false,
            color: colors ? 0xffffff : 0x00ff00
        }});
        
        const pointCloud = new THREE.Points(geometry, material);
        scene.add(pointCloud);
        
        // Controls
        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        
        // Position camera
        const box = new THREE.Box3().setFromObject(pointCloud);
        const center = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z);
        
        camera.position.set(center.x + maxDim, center.y + maxDim, center.z + maxDim);
        camera.lookAt(center);
        controls.target.copy(center);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            renderer.render(scene, camera);
        }}
        
        animate();
        
        // Handle resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
    </script>
</body>
</html>
"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        return output_path
    
    def _visualize_web(self,
                      points_3d: np.ndarray,
                      colors: Optional[np.ndarray],
                      camera_poses: Optional[Dict[int, np.ndarray]],
                      title: str) -> str:
        """Create web-based viewer with SSH tunnel instructions."""
        
        # First export the HTML viewer
        html_path = self._create_simple_html_viewer(points_3d, colors, f"{title.lower().replace(' ', '_')}_web.html")
        
        # Provide SSH tunnel instructions
        logging.info("=" * 60)
        logging.info("WEB VIEWER WITH SSH TUNNEL")
        logging.info("=" * 60)
        logging.info("To view the web-based 3D viewer over SSH:")
        logging.info("")
        logging.info("1. On your LOCAL machine, create an SSH tunnel:")
        logging.info(f"   ssh -L 8080:localhost:8080 your_username@your_server")
        logging.info("")
        logging.info("2. On the SERVER (this machine), start a simple web server:")
        logging.info(f"   cd {self.output_dir}")
        logging.info("   python -m http.server 8080")
        logging.info("")
        logging.info("3. Open your LOCAL browser and go to:")
        logging.info("   http://localhost:8080/" + html_path.name)
        logging.info("")
        logging.info("=" * 60)
        
        return str(html_path)
    
    def _visualize_terminal(self,
                          points_3d: np.ndarray,
                          colors: Optional[np.ndarray],
                          camera_poses: Optional[Dict[int, np.ndarray]],
                          title: str) -> str:
        """Create ASCII-based terminal visualization for quick preview."""
        
        logging.info("=" * 60)
        logging.info(f"TERMINAL VISUALIZATION: {title}")
        logging.info("=" * 60)
        
        # Basic statistics
        logging.info(f"Number of points: {len(points_3d)}")
        
        if len(points_3d) > 0:
            # Compute bounding box
            min_coords = np.min(points_3d, axis=0)
            max_coords = np.max(points_3d, axis=0)
            center = np.mean(points_3d, axis=0)
            
            logging.info(f"Bounding box:")
            logging.info(f"  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]")
            logging.info(f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]")
            logging.info(f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]")
            logging.info(f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
            
            # Create simple ASCII projection (XY plane)
            self._create_ascii_projection(points_3d, "XY Projection (Top View)")
        
        if camera_poses:
            logging.info(f"Number of cameras: {len(camera_poses)}")
            for cam_id, pose in camera_poses.items():
                if pose.shape == (3, 4):
                    R = pose[:3, :3]
                    t = pose[:3, 3]
                    center = -R.T @ t
                    logging.info(f"  Camera {cam_id}: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
        
        logging.info("=" * 60)
        
        # Also save a simple text summary
        output_path = self.output_dir / f"{title.lower().replace(' ', '_')}_summary.txt"
        with open(output_path, 'w') as f:
            f.write(f"SfM Visualization Summary: {title}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Number of points: {len(points_3d)}\n")
            if len(points_3d) > 0:
                f.write(f"Bounding box:\n")
                f.write(f"  X: [{min_coords[0]:.3f}, {max_coords[0]:.3f}]\n")
                f.write(f"  Y: [{min_coords[1]:.3f}, {max_coords[1]:.3f}]\n")
                f.write(f"  Z: [{min_coords[2]:.3f}, {max_coords[2]:.3f}]\n")
                f.write(f"Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n")
            
            if camera_poses:
                f.write(f"\nNumber of cameras: {len(camera_poses)}\n")
                for cam_id, pose in camera_poses.items():
                    if pose.shape == (3, 4):
                        R = pose[:3, :3]
                        t = pose[:3, 3]
                        center = -R.T @ t
                        f.write(f"  Camera {cam_id}: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})\n")
        
        return str(output_path)
    
    def _create_ascii_projection(self, points_3d: np.ndarray, title: str):
        """Create ASCII art projection of point cloud."""
        
        if len(points_3d) == 0:
            return
        
        # Project to XY plane
        x_coords = points_3d[:, 0]
        y_coords = points_3d[:, 1]
        
        # Normalize to terminal size
        width, height = 60, 20
        
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        
        if x_max == x_min or y_max == y_min:
            logging.info("Cannot create ASCII projection - points are collinear")
            return
        
        # Map to grid
        x_grid = ((x_coords - x_min) / (x_max - x_min) * (width - 1)).astype(int)
        y_grid = ((y_coords - y_min) / (y_max - y_min) * (height - 1)).astype(int)
        
        # Create ASCII grid
        grid = [[' ' for _ in range(width)] for _ in range(height)]
        
        for x, y in zip(x_grid, y_grid):
            if 0 <= x < width and 0 <= y < height:
                grid[height - 1 - y][x] = '*'  # Flip Y for proper orientation
        
        # Print ASCII art
        logging.info(f"\n{title}:")
        logging.info("+" + "-" * width + "+")
        for row in grid:
            logging.info("|" + "".join(row) + "|")
        logging.info("+" + "-" * width + "+")
    
    def _error_to_color(self, errors: np.ndarray) -> np.ndarray:
        """Convert reprojection errors to color map (green=good, red=bad)."""
        # Normalize errors to [0, 1]
        max_error = np.percentile(errors, 95)  # Use 95th percentile to avoid outliers
        if max_error == 0:
            max_error = 1.0
        normalized_errors = np.clip(errors / max_error, 0, 1)
        
        # Create color map: green (low error) to red (high error)
        colors = np.zeros((len(errors), 3))
        colors[:, 0] = normalized_errors  # Red channel
        colors[:, 1] = 1 - normalized_errors  # Green channel
        colors[:, 2] = 0.2  # Small blue component for visibility
        
        return colors
    
    def _print_viewing_instructions(self, ply_path: Path, obj_path: Path, html_path: Path):
        """Print instructions for viewing exported files."""
        
        logging.info("\n" + "=" * 60)
        logging.info("VIEWING INSTRUCTIONS")
        logging.info("=" * 60)
        
        logging.info("\n1. DOWNLOAD AND VIEW LOCALLY:")
        logging.info("   Use scp to download files to your local machine:")
        logging.info(f"   scp your_server:{ply_path} .")
        logging.info(f"   scp your_server:{html_path} .")
        logging.info("")
        logging.info("   Then view with:")
        logging.info("   - PLY files: MeshLab, CloudCompare, Blender")
        logging.info("   - HTML file: Any web browser (double-click)")
        
        logging.info("\n2. SSH X11 FORWARDING:")
        logging.info("   Connect with: ssh -X your_server")
        logging.info("   Then run the visualization again with method='matplotlib'")
        
        logging.info("\n3. SSH TUNNEL FOR WEB VIEWER:")
        logging.info("   Local: ssh -L 8080:localhost:8080 your_server")
        logging.info(f"   Server: cd {self.output_dir.parent} && python -m http.server 8080")
        logging.info("   Browser: http://localhost:8080/visualizations/" + html_path.name)
        
        logging.info("\n" + "=" * 60)


def create_visualizer(config: Dict) -> SfMVisualizer:
    """Factory function to create SfM visualizer."""
    return SfMVisualizer(config)


if __name__ == "__main__":
    # Example usage
    import yaml
    
    # Load configuration
    with open("../../config/base_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Create visualizer
    visualizer = create_visualizer(config)
    
    print("SfMVisualizer initialized successfully!")
    print("Available methods: auto, matplotlib, export, web, terminal")
