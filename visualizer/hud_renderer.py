"""
VisionSpace - Real-Time OpenCV HUD & Multi-View Dashboard Renderer
Composes a high-tech head-up display combining RGB object detections,
colorized depth maps, top-down spatial radar, and semantic telemetry readouts.
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Tuple


class HUDRenderer:
    """Renders comprehensive multi-pane computer vision dashboard."""

    def __init__(self, target_width: int = 1280, target_height: int = 720):
        self.target_width = target_width
        self.target_height = target_height
        self._fps_history = []
        self._last_time = time.time()

    def update_fps(self) -> float:
        """Calculate smoothed real-time frames per second."""
        current_time = time.time()
        dt = current_time - self._last_time
        self._last_time = current_time
        fps = 1.0 / dt if dt > 0 else 30.0
        self._fps_history.append(fps)
        if len(self._fps_history) > 15:
            self._fps_history.pop(0)
        return float(np.mean(self._fps_history))

    def draw_spatial_overlay(self, frame: np.ndarray,
                             spatial_objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Draws 3D spatial annotations, bounding boxes, and distance tags on RGB frame.
        """
        overlay = frame.copy()

        for obj in spatial_objects:
            bbox = obj["bbox"]
            x1, y1, x2, y2 = bbox
            pos = obj["position_3d"]
            cls_name = obj["class_name"]
            dist = obj["distance_m"]
            bearing = obj["bearing_deg"]
            status = obj["proximity_status"]
            color = obj["status_color"]

            # Draw 2D bounding box with rounded corner accents
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Draw crosshair at center
            cx, cy = obj["center"]
            cv2.drawMarker(overlay, (cx, cy), color, cv2.MARKER_CROSS, 10, 1)

            # Badge banner with Class, 3D metric distance and bearing
            tag = f"{cls_name.upper()} | {dist}m | {pos['x']:+.1f}m X"
            if abs(bearing) > 5:
                tag += f" ({bearing:+.0f} deg)"

            (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(overlay, (x1, max(0, y1 - 22)), (x1 + tw + 10, y1), color, -1)
            cv2.putText(overlay, tag, (x1 + 5, y1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)

            # Proximity warning pill if critical
            if status == "CRITICAL":
                cv2.putText(overlay, "! CLOSE !", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        return overlay

    def render_dashboard(self, rgb_annotated: np.ndarray,
                         depth_colormap: np.ndarray,
                         bev_radar: np.ndarray,
                         semantic_info: Dict[str, Any],
                         fps: float) -> np.ndarray:
        """
        Assembles all visual components into a unified 1280x720 professional dashboard:
        - Top-Left: RGB Stream with Spatial Detections (640x480)
        - Top-Right: MiDaS Dense Depth Map (640x480)
        - Bottom-Left: Top-down BEV Spatial Radar Map
        - Bottom-Right: Semantic Telemetry & Situational Awareness Panel
        """
        canvas = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)
        canvas[:] = (15, 18, 22)  # High-tech dark theme

        # Top Bar (Header)
        cv2.rectangle(canvas, (0, 0), (self.target_width, 42), (25, 30, 40), -1)
        cv2.line(canvas, (0, 42), (self.target_width, 42), (0, 210, 255), 2)
        cv2.putText(canvas, "VISIONSPACE : REAL-TIME 3D SCENE UNDERSTANDING & SPATIAL COMPUTING",
                    (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"FPS: {fps:.1f} | ENGINE: YOLOv11 + MiDaS + Open3D",
                    (self.target_width - 480, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # Upper Left: RGB Feed (Resize to 630x350)
        feed_w, feed_h = 630, 350
        rgb_resized = cv2.resize(rgb_annotated, (feed_w, feed_h))
        canvas[50:50 + feed_h, 10:10 + feed_w] = rgb_resized
        cv2.rectangle(canvas, (10, 50), (10 + feed_w, 50 + feed_h), (50, 60, 75), 1)
        cv2.putText(canvas, "RGB FEED & SPATIAL 3D BOUNDING", (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Upper Right: Depth Colormap (Resize to 630x350)
        depth_resized = cv2.resize(depth_colormap, (feed_w, feed_h))
        canvas[50:50 + feed_h, 640:640 + feed_w] = depth_resized
        cv2.rectangle(canvas, (640, 50), (640 + feed_w, 50 + feed_h), (50, 60, 75), 1)
        cv2.putText(canvas, "MiDaS MONOCULAR DEPTH ESTIMATION", (650, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Lower Left: Top-Down BEV Radar (300x300 resized into lower left area)
        radar_h = 300
        radar_w = 300
        radar_resized = cv2.resize(bev_radar, (radar_w, radar_h))
        canvas[410:410 + radar_h, 10:10 + radar_w] = radar_resized
        cv2.rectangle(canvas, (10, 410), (10 + radar_w, 410 + radar_h), (50, 60, 75), 1)

        # Lower Center/Right: Semantic Scene Understanding & Telemetry (width: 950px)
        telemetry_x = 320
        telemetry_y = 410
        telemetry_w = self.target_width - telemetry_x - 10
        telemetry_h = 300

        # Draw Telemetry Card
        cv2.rectangle(canvas, (telemetry_x, telemetry_y),
                      (telemetry_x + telemetry_w, telemetry_y + telemetry_h), (22, 26, 34), -1)
        cv2.rectangle(canvas, (telemetry_x, telemetry_y),
                      (telemetry_x + telemetry_w, telemetry_y + telemetry_h), (50, 60, 75), 1)

        # Telemetry Card Header
        cv2.putText(canvas, "SEMANTIC SCENE UNDERSTANDING & SPATIAL TELEMETRY",
                    (telemetry_x + 15, telemetry_y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.line(canvas, (telemetry_x + 15, telemetry_y + 35),
                 (telemetry_x + telemetry_w - 15, telemetry_y + 35), (45, 55, 70), 1)

        # Scene Category
        scene_type = semantic_info.get("scene_type", "Unknown")
        cv2.putText(canvas, f"Scene Classification: {scene_type}",
                    (telemetry_x + 15, telemetry_y + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Detected Object Count
        obj_count = semantic_info.get("object_count", 0)
        cv2.putText(canvas, f"Active Spatial Trackers: {obj_count}",
                    (telemetry_x + 15, telemetry_y + 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 200, 220), 1, cv2.LINE_AA)

        # Hazard Warning Banner
        hazards = semantic_info.get("hazard_warnings", [])
        if hazards:
            cv2.rectangle(canvas, (telemetry_x + 15, telemetry_y + 115),
                          (telemetry_x + telemetry_w - 15, telemetry_y + 150), (0, 0, 160), -1)
            cv2.putText(canvas, hazards[0], (telemetry_x + 25, telemetry_y + 138),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(canvas, "Path Safety: CLEAR (No critical proximity hazards)",
                        (telemetry_x + 15, telemetry_y + 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 120), 1, cv2.LINE_AA)

        # Natural Language Summary
        summary = semantic_info.get("summary", "")
        # Wrap summary text across two lines
        cv2.putText(canvas, "Spatial Situational Narrative:",
                    (telemetry_x + 15, telemetry_y + 175),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)

        words = summary.split(" ")
        line1 = " ".join(words[:14])
        line2 = " ".join(words[14:28])
        cv2.putText(canvas, line1, (telemetry_x + 15, telemetry_y + 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)
        if line2:
            cv2.putText(canvas, line2, (telemetry_x + 15, telemetry_y + 225),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)

        # Spatial Relationships
        relations = semantic_info.get("relationships", [])
        rel_str = " | ".join(relations[:2]) if relations else "Isolated object spatial arrangements."
        cv2.putText(canvas, f"Spatial Topology: {rel_str[:85]}",
                    (telemetry_x + 15, telemetry_y + 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (150, 170, 190), 1, cv2.LINE_AA)

        # Footer hotkeys
        cv2.putText(canvas, "[Q]: Quit  |  [S]: Save 3D Point Cloud (.PLY)  |  [W]: Open 3D Web Viewer  |  [V]: Open3D 3D Window",
                    (telemetry_x + 15, telemetry_y + 288),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 210, 255), 1, cv2.LINE_AA)

        return canvas
