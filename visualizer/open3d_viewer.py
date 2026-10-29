"""
VisionSpace - Open3D Desktop Interactive 3D Visualizer
Displays the dense 3D point cloud and 3D object bounding boxes in an interactive Open3D window.
"""

import open3d as o3d
import numpy as np
from typing import List, Dict, Any, Optional


class Open3DViewer:
    """Manages Open3D desktop 3D rendering and geometry displays."""

    def __init__(self, window_name: str = "VisionSpace - Interactive 3D Scene"):
        self.window_name = window_name

    def display_scene(self, pcd: o3d.geometry.PointCloud,
                      spatial_objects: Optional[List[Dict[str, Any]]] = None):
        """
        Opens an interactive Open3D 3D viewer showing the colored point cloud
        and 3D bounding boxes with coordinate axes.
        """
        geometries = []

        # Add coordinate frame at camera origin (X=Red, Y=Green, Z=Blue)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        geometries.append(coord_frame)

        # Add Point Cloud
        if pcd is not None and len(pcd.points) > 0:
            geometries.append(pcd)

        # Add 3D Oriented Bounding Boxes around detected objects
        if spatial_objects:
            for obj in spatial_objects:
                pos = obj["position_3d"]
                dims = obj["dimensions_3d"]
                color = [c / 255.0 for c in reversed(obj["status_color"])]

                center = np.array([pos["x"], -pos["y"], pos["z"]])
                extent = np.array([
                    max(0.1, dims["width"]),
                    max(0.1, dims["height"]),
                    max(0.1, dims["depth"])
                ])

                bbox_3d = o3d.geometry.OrientedBoundingBox(center, np.identity(3), extent)
                bbox_3d.color = color
                geometries.append(bbox_3d)

        print(f"[VisionSpace] Opening Open3D interactive viewer with {len(geometries)} geometries...")
        print("[VisionSpace] Controls: Left-Click+Drag: Rotate | Shift+Left-Click: Pan | Wheel: Zoom")
        o3d.visualization.draw_geometries(
            geometries,
            window_name=self.window_name,
            width=1024,
            height=768,
            left=50,
            top=50
        )
