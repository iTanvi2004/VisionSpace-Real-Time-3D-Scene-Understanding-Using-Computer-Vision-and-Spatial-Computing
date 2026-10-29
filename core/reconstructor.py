"""
VisionSpace - 3D Scene Reconstruction & Point Cloud Generator
Reconstructs dense 3D colored point clouds from RGB-D inputs,
performs voxel downsampling, statistical outlier removal, normal estimation via Open3D,
and exports 3D representations (.ply, .obj, JSON).
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import open3d as o3d
from config.settings import (
    FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
    PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y,
    POINT_CLOUD_VOXEL_SIZE, POINT_CLOUD_MAX_POINTS,
    OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO,
    OUTPUTS_DIR
)


class SceneReconstructor3D:
    """Creates, filters, and exports 3D Point Cloud representations of the scene."""

    def __init__(self, fx: float = FOCAL_LENGTH_X, fy: float = FOCAL_LENGTH_Y,
                 cx: float = PRINCIPAL_POINT_X, cy: float = PRINCIPAL_POINT_Y):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def build_point_cloud(self, frame_bgr: np.ndarray, depth_meters: np.ndarray,
                          stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Constructs 3D XYZ coordinates and RGB color arrays from RGB image and depth map.

        Returns:
            points_xyz: (N, 3) float32 array
            colors_rgb: (N, 3) float32 array in [0, 1] range
        """
        h, w = depth_meters.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Generate pixel grid with specified stride for efficiency
        u_coords = np.arange(0, w, stride)
        v_coords = np.arange(0, h, stride)
        uu, vv = np.meshgrid(u_coords, v_coords)

        z = depth_meters[vv, uu]

        # Valid depth mask (filter extreme outliers)
        valid_mask = (z > 0.1) & (z < 15.0) & (~np.isnan(z)) & (~np.isinf(z))

        uu_valid = uu[valid_mask]
        vv_valid = vv[valid_mask]
        z_valid = z[valid_mask]

        # 3D back-projection (Pinhole camera model)
        x_valid = ((uu_valid - self.cx) * z_valid) / self.fx
        # Flip Y to follow standard 3D world convention (Y is UP)
        y_valid = -((vv_valid - self.cy) * z_valid) / self.fy

        points_xyz = np.stack([x_valid, y_valid, z_valid], axis=-1).astype(np.float32)

        # Sample colors
        colors = frame_rgb[vv_valid, uu_valid].astype(np.float32) / 255.0

        # Subsample if exceeds max point threshold
        if len(points_xyz) > POINT_CLOUD_MAX_POINTS:
            idx = np.random.choice(len(points_xyz), POINT_CLOUD_MAX_POINTS, replace=False)
            points_xyz = points_xyz[idx]
            colors = colors[idx]

        return points_xyz, colors

    def create_open3d_point_cloud(self, points_xyz: np.ndarray,
                                  colors_rgb: np.ndarray,
                                  voxel_size: float = POINT_CLOUD_VOXEL_SIZE,
                                  remove_outliers: bool = True) -> o3d.geometry.PointCloud:
        """
        Constructs an Open3D PointCloud object with voxel downsampling and filtering.
        """
        pcd = o3d.geometry.PointCloud()
        if len(points_xyz) == 0:
            return pcd

        pcd.points = o3d.utility.Vector3dVector(points_xyz.astype(np.float64))
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64))

        # Voxel grid downsampling
        if voxel_size > 0:
            pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        # Statistical outlier removal
        if remove_outliers and len(pcd.points) > OUTLIER_NB_NEIGHBORS:
            pcd, _ = pcd.remove_statistical_outlier(
                nb_neighbors=OUTLIER_NB_NEIGHBORS,
                std_ratio=OUTLIER_STD_RATIO
            )

        # Estimate surface normals
        if len(pcd.points) > 10:
            pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
            )

        return pcd

    def export_point_cloud(self, pcd: o3d.geometry.PointCloud,
                           filename: str = "scene_reconstruction.ply") -> str:
        """
        Saves the reconstructed 3D point cloud to disk (.ply or .pcd).
        """
        save_path = OUTPUTS_DIR / filename
        o3d.io.write_point_cloud(str(save_path), pcd, write_ascii=False)
        print(f"[VisionSpace] 3D Point Cloud exported successfully: {save_path}")
        return str(save_path)

    def to_web_json(self, points_xyz: np.ndarray, colors_rgb: np.ndarray,
                    max_points: int = 15000) -> Dict[str, Any]:
        """
        Serializes downsampled 3D point cloud into compact JSON payload for WebGL / Three.js.
        """
        n = len(points_xyz)
        if n == 0:
            return {"points": [], "colors": []}

        if n > max_points:
            indices = np.random.choice(n, max_points, replace=False)
            pts = points_xyz[indices]
            cls = colors_rgb[indices]
        else:
            pts = points_xyz
            cls = colors_rgb

        # Flatten into interleaved lists for fast browser WebGL buffer streaming
        flat_pts = np.round(pts, 3).flatten().tolist()
        flat_cls = np.round(cls, 3).flatten().tolist()

        return {
            "point_count": len(pts),
            "points": flat_pts,
            "colors": flat_cls
        }
