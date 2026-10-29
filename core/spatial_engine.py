"""
VisionSpace - Spatial Computing & Feature Extraction Engine
Performs 3D pinhole camera back-projection, metric 3D centroid and bounding box computation,
proximity hazard risk assessment, and top-down Bird's Eye View (BEV) radar mapping.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from config.settings import (
    FOCAL_LENGTH_X, FOCAL_LENGTH_Y,
    PRINCIPAL_POINT_X, PRINCIPAL_POINT_Y,
    DISTANCE_THRESHOLD_CRITICAL, DISTANCE_THRESHOLD_WARNING
)


class SpatialEngine:
    """Computes 3D coordinates, spatial bounding boxes, and top-down radar representations."""

    def __init__(self, fx: float = FOCAL_LENGTH_X, fy: float = FOCAL_LENGTH_Y,
                 cx: float = PRINCIPAL_POINT_X, cy: float = PRINCIPAL_POINT_Y):
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy

    def pixel_to_3d(self, u: float, v: float, z: float) -> Tuple[float, float, float]:
        """
        Back-projects a 2D image pixel (u, v) with depth z into 3D camera coordinates (X, Y, Z).
        Coordinate System:
          +X: Right
          +Y: Down (or up depending on convention; here camera frame Y is down)
          +Z: Forward (away from camera)
        """
        x = ((u - self.cx) * z) / self.fx
        y = ((v - self.cy) * z) / self.fy
        return round(float(x), 3), round(float(y), 3), round(float(z), 3)

    def extract_spatial_features(self, detections: List[Dict[str, Any]],
                                 depth_meters: np.ndarray) -> List[Dict[str, Any]]:
        """
        Enriches 2D detections with 3D spatial properties:
        - 3D centroid (X, Y, Z) in meters
        - 3D estimated dimensions (width_m, height_m, depth_m)
        - Proximity status ('CRITICAL', 'WARNING', 'SAFE')
        - Bearing angle (degrees relative to optical axis)
        """
        spatial_objects = []

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox
            cx, cy = det["center"]

            # Estimate distance Z
            # Sample median of object bounding box
            z = self._sample_median_depth(depth_meters, bbox)

            # Backproject centroid into 3D space
            X, Y, Z = self.pixel_to_3d(cx, cy, z)

            # Estimate 3D physical width and height
            x_min, _, _ = self.pixel_to_3d(x1, cy, z)
            x_max, _, _ = self.pixel_to_3d(x2, cy, z)
            _, y_min, _ = self.pixel_to_3d(cx, y1, z)
            _, y_max, _ = self.pixel_to_3d(cx, y2, z)

            width_m = round(abs(x_max - x_min), 2)
            height_m = round(abs(y_max - y_min), 2)
            # Approximate object depth extent proportionally
            depth_m = round(min(width_m, height_m) * 0.8, 2)

            # Compute horizontal bearing angle from optical center (degrees)
            # 0 deg = straight ahead, negative = left, positive = right
            bearing_rad = np.arctan2(X, Z)
            bearing_deg = round(float(np.degrees(bearing_rad)), 1)

            # Total Euclidean distance
            euclidean_dist = round(float(np.sqrt(X**2 + Y**2 + Z**2)), 2)

            # Proximity status
            if euclidean_dist < DISTANCE_THRESHOLD_CRITICAL:
                status = "CRITICAL"
                color = (0, 0, 255)      # Red
            elif euclidean_dist < DISTANCE_THRESHOLD_WARNING:
                status = "WARNING"
                color = (0, 200, 255)    # Yellow/Orange
            else:
                status = "SAFE"
                color = (0, 255, 0)      # Green

            # 3D bounding box corners in camera frame
            corners_3d = self._compute_3d_box_corners(X, Y, Z, width_m, height_m, depth_m)

            spatial_obj = {
                **det,
                "position_3d": {"x": X, "y": Y, "z": Z},
                "dimensions_3d": {"width": width_m, "height": height_m, "depth": depth_m},
                "distance_m": euclidean_dist,
                "bearing_deg": bearing_deg,
                "proximity_status": status,
                "status_color": color,
                "corners_3d": corners_3d
            }
            spatial_objects.append(spatial_obj)

        # Sort from closest to furthest
        spatial_objects.sort(key=lambda item: item["distance_m"])
        return spatial_objects

    def _sample_median_depth(self, depth_meters: np.ndarray, bbox: list) -> float:
        """Sample robust depth from the inner portion of the bounding box."""
        h, w = depth_meters.shape[:2]
        x1 = max(0, min(bbox[0], w - 1))
        x2 = max(0, min(bbox[1], w))
        y1 = max(0, min(bbox[2], h - 1))
        y2 = max(0, min(bbox[3], h))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        cx1 = int(x1 + 0.20 * bw)
        cx2 = int(x2 - 0.20 * bw)
        cy1 = int(y1 + 0.20 * bh)
        cy2 = int(y2 - 0.20 * bh)

        roi = depth_meters[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            roi = depth_meters[y1:y2, x1:x2]

        if roi.size == 0:
            return 2.5

        return float(np.median(roi))

    def _compute_3d_box_corners(self, x: float, y: float, z: float,
                                w: float, h: float, d: float) -> List[List[float]]:
        """Compute the 8 vertices of the 3D bounding box."""
        hw = w / 2.0
        hh = h / 2.0
        hd = d / 2.0

        corners = [
            [x - hw, y - hh, z - hd],
            [x + hw, y - hh, z - hd],
            [x + hw, y + hh, z - hd],
            [x - hw, y + hh, z - hd],
            [x - hw, y - hh, z + hd],
            [x + hw, y - hh, z + hd],
            [x + hw, y + hh, z + hd],
            [x - hw, y + hh, z + hd],
        ]
        return [[round(coord, 3) for coord in c] for c in corners]

    def generate_bev_radar(self, spatial_objects: List[Dict[str, Any]],
                           radar_size: int = 300, max_range_m: float = 6.0) -> np.ndarray:
        """
        Renders a Top-Down Bird's Eye View (BEV) radar map.
        X-axis (horizontal): Left (-) to Right (+) in meters
        Z-axis (vertical): Forward distance from camera (0 to max_range_m)
        """
        radar = np.zeros((radar_size, radar_size, 3), dtype=np.uint8)
        radar[:] = (20, 22, 28)  # Sleek dark navy slate background

        origin_x = radar_size // 2
        origin_y = radar_size - 25

        # Scale factor (pixels per meter)
        ppm = (radar_size - 40) / max_range_m

        # Draw range circles (1m, 2m, 3m, 4m, 5m)
        for r_m in range(1, int(max_range_m) + 1):
            r_px = int(r_m * ppm)
            cv2.circle(radar, (origin_x, origin_y), r_px, (45, 50, 60), 1, cv2.LINE_AA)
            cv2.putText(radar, f"{r_m}m", (origin_x + 5, origin_y - r_px + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (100, 110, 130), 1)

        # Draw Field of View (FOV) cone (~60 degrees)
        fov_angle_rad = np.radians(35)
        fov_len_px = int(max_range_m * ppm)
        left_x = int(origin_x - fov_len_px * np.sin(fov_angle_rad))
        left_y = int(origin_y - fov_len_px * np.cos(fov_angle_rad))
        right_x = int(origin_x + fov_len_px * np.sin(fov_angle_rad))
        right_y = int(origin_y - fov_len_px * np.cos(fov_angle_rad))

        cv2.line(radar, (origin_x, origin_y), (left_x, left_y), (60, 70, 85), 1, cv2.LINE_AA)
        cv2.line(radar, (origin_x, origin_y), (right_x, right_y), (60, 70, 85), 1, cv2.LINE_AA)
        cv2.line(radar, (origin_x, origin_y), (origin_x, origin_y - fov_len_px), (40, 50, 65), 1, cv2.LINE_AA)

        # Plot camera marker (Cyan triangle)
        cam_pts = np.array([
            [origin_x, origin_y - 8],
            [origin_x - 7, origin_y + 6],
            [origin_x + 7, origin_y + 6]
        ], np.int32)
        cv2.fillPoly(radar, [cam_pts], (255, 200, 0))

        # Plot each detected object on BEV map
        for obj in spatial_objects:
            pos = obj["position_3d"]
            obj_x = pos["x"]
            obj_z = pos["z"]
            color = obj["status_color"]
            label = obj["class_name"]
            dist = obj["distance_m"]

            # Map to radar pixel coordinates
            px = int(origin_x + (obj_x * ppm))
            py = int(origin_y - (obj_z * ppm))

            # Clamp within radar bounds
            if 10 <= px < radar_size - 10 and 10 <= py < radar_size - 10:
                # Draw object point and halo
                cv2.circle(radar, (px, py), 7, color, -1, cv2.LINE_AA)
                cv2.circle(radar, (px, py), 10, (255, 255, 255), 1, cv2.LINE_AA)

                # Label text
                tag = f"{label[:4]} {dist}m"
                cv2.putText(radar, tag, (px + 9, py + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1)

        # Header title
        cv2.putText(radar, "BEV SPATIAL RADAR", (10, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)

        return radar
