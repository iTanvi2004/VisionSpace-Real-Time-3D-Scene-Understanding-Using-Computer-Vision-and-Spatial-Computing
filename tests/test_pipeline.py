"""
VisionSpace - Comprehensive Unit and Integration Test Suite
Tests each module individually and runs an end-to-end multi-frame integration test.
"""

import sys
import unittest
import numpy as np
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import FRAME_WIDTH, FRAME_HEIGHT, OUTPUTS_DIR
from core.camera import CameraStream
from core.detector import ObjectDetector
from core.depth_estimator import DepthEstimator
from core.spatial_engine import SpatialEngine
from core.scene_understanding import SemanticSceneUnderstander
from core.reconstructor import SceneReconstructor3D
from visualizer.hud_renderer import HUDRenderer
from samples.generate_sample import generate_benchmark_room_image


class TestVisionSpacePipeline(unittest.TestCase):
    """Test suite covering all VisionSpace pipeline components."""

    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  Running VisionSpace Perception Test Suite")
        print("=======================================================")
        cls.sample_img_path = generate_benchmark_room_image()
        cls.camera = CameraStream(source="synthetic", width=FRAME_WIDTH, height=FRAME_HEIGHT)
        cls.detector = ObjectDetector()
        cls.depth_estimator = DepthEstimator()
        cls.spatial_engine = SpatialEngine()
        cls.scene_understander = SemanticSceneUnderstander()
        cls.reconstructor = SceneReconstructor3D()
        cls.hud_renderer = HUDRenderer()

    def test_01_camera_stream(self):
        """Test CameraStream synthetic and image frame generation."""
        ret, frame = self.camera.read()
        self.assertTrue(ret, "Camera stream should successfully return a frame.")
        self.assertIsNotNone(frame, "Frame should not be None.")
        self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3), "Frame dimensions must match config.")
        print("  [PASS] test_01_camera_stream")

    def test_02_object_detector(self):
        """Test YOLOv11 object detection on synthetic room frame."""
        ret, frame = self.camera.read()
        self.assertTrue(ret)
        detections = self.detector.detect(frame)
        self.assertIsInstance(detections, list, "Detections must be a list.")
        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("class_name", det)
            self.assertIn("confidence", det)
            self.assertIn("center", det)
            self.assertGreaterEqual(det["confidence"], 0.2)
        print(f"  [PASS] test_02_object_detector (Detected {len(detections)} objects)")

    def test_03_depth_estimator(self):
        """Test MiDaS depth estimation output shapes and metrics."""
        ret, frame = self.camera.read()
        self.assertTrue(ret)
        depth_m, depth_cmap = self.depth_estimator.estimate(frame)

        self.assertEqual(depth_m.shape, (FRAME_HEIGHT, FRAME_WIDTH), "Depth map shape must match image.")
        self.assertEqual(depth_cmap.shape, (FRAME_HEIGHT, FRAME_WIDTH, 3), "Colormap must be 3-channel BGR.")
        self.assertTrue(np.all(depth_m > 0.0), "Estimated depth in meters must be positive.")
        self.assertFalse(np.isnan(depth_m).any(), "Depth map must not contain NaN.")
        print(f"  [PASS] test_03_depth_estimator (Depth range: {depth_m.min():.2f}m - {depth_m.max():.2f}m)")

    def test_04_spatial_engine(self):
        """Test 3D back-projection, proximity risk, and BEV radar."""
        ret, frame = self.camera.read()
        self.assertTrue(ret)
        depth_m, _ = self.depth_estimator.estimate(frame)
        detections = self.detector.detect(frame)

        # Ensure at least one dummy detection if detector finds none on synthetic pattern
        if not detections:
            detections = [{
                "bbox": [250, 200, 390, 340],
                "center": (320, 270),
                "width": 140, "height": 140,
                "class_id": 63, "class_name": "laptop", "confidence": 0.85
            }]

        spatial_objs = self.spatial_engine.extract_spatial_features(detections, depth_m)
        self.assertGreaterEqual(len(spatial_objs), 1)

        first_obj = spatial_objs[0]
        self.assertIn("position_3d", first_obj)
        self.assertIn("distance_m", first_obj)
        self.assertIn("proximity_status", first_obj)
        self.assertIn(first_obj["proximity_status"], ["CRITICAL", "WARNING", "SAFE"])

        # Test BEV Radar generation
        radar = self.spatial_engine.generate_bev_radar(spatial_objs, radar_size=300)
        self.assertEqual(radar.shape, (300, 300, 3), "Radar map must be 300x300x3.")
        print(f"  [PASS] test_04_spatial_engine (3D pos: {first_obj['position_3d']})")

    def test_05_scene_understanding(self):
        """Test semantic scene reasoning, relations, and summary."""
        ret, frame = self.camera.read()
        depth_m, _ = self.depth_estimator.estimate(frame)
        detections = self.detector.detect(frame)
        if not detections:
            detections = [
                {"bbox": [250, 200, 390, 340], "center": (320, 270), "width": 140, "height": 140,
                 "class_id": 63, "class_name": "laptop", "confidence": 0.85},
                {"bbox": [440, 240, 480, 290], "center": (460, 265), "width": 40, "height": 50,
                 "class_id": 41, "class_name": "cup", "confidence": 0.75}
            ]

        spatial_objs = self.spatial_engine.extract_spatial_features(detections, depth_m)
        semantic_info = self.scene_understander.understand_scene(spatial_objs)

        self.assertIn("scene_type", semantic_info)
        self.assertIn("summary", semantic_info)
        self.assertIn("hazard_warnings", semantic_info)
        self.assertIn("relationships", semantic_info)
        self.assertGreater(len(semantic_info["summary"]), 10)
        print(f"  [PASS] test_05_scene_understanding ('{semantic_info['scene_type']}')")

    def test_06_3d_reconstruction_and_export(self):
        """Test point cloud building, Open3D conversion, and PLY export."""
        ret, frame = self.camera.read()
        depth_m, _ = self.depth_estimator.estimate(frame)

        pts, colors = self.reconstructor.build_point_cloud(frame, depth_m, stride=8)
        self.assertGreater(len(pts), 100, "Point cloud should contain valid 3D points.")
        self.assertEqual(pts.shape[1], 3, "Points must be 3D (X, Y, Z).")
        self.assertEqual(colors.shape[1], 3, "Colors must be 3-channel (R, G, B).")

        # Open3D point cloud
        pcd = self.reconstructor.create_open3d_point_cloud(pts, colors, voxel_size=0.05)
        self.assertGreater(len(pcd.points), 0)

        # Export test PLY
        test_ply = "test_output.ply"
        saved_path = self.reconstructor.export_point_cloud(pcd, test_ply)
        self.assertTrue(Path(saved_path).exists(), "Exported PLY file must exist on disk.")

        # Test Web JSON conversion
        web_json = self.reconstructor.to_web_json(pts, colors, max_points=1000)
        self.assertIn("points", web_json)
        self.assertIn("colors", web_json)
        self.assertGreater(len(web_json["points"]), 0)
        print(f"  [PASS] test_06_3d_reconstruction_and_export (Points: {len(pcd.points)})")

    def test_07_hud_renderer(self):
        """Test full multi-view HUD dashboard assembly."""
        ret, frame = self.camera.read()
        depth_m, depth_cmap = self.depth_estimator.estimate(frame)
        detections = self.detector.detect(frame)
        spatial_objs = self.spatial_engine.extract_spatial_features(detections, depth_m)
        bev_radar = self.spatial_engine.generate_bev_radar(spatial_objs)
        semantic_info = self.scene_understander.understand_scene(spatial_objs)
        fps = self.hud_renderer.update_fps()
        rgb_annotated = self.hud_renderer.draw_spatial_overlay(frame, spatial_objs)

        dashboard = self.hud_renderer.render_dashboard(
            rgb_annotated=rgb_annotated,
            depth_colormap=depth_cmap,
            bev_radar=bev_radar,
            semantic_info=semantic_info,
            fps=fps
        )
        self.assertEqual(dashboard.shape, (720, 1280, 3), "Dashboard must be 1280x720x3.")
        print("  [PASS] test_07_hud_renderer (Dashboard 1280x720 rendered)")


if __name__ == "__main__":
    unittest.main()
