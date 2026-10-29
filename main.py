"""
VisionSpace: Real-Time 3D Scene Understanding Using Computer Vision and Spatial Computing
Department of Computer Science & Engineering (AIML) - Goel Institute of Technology & Management

Main Pipeline Runner:
Integrates YOLOv11 object detection, MiDaS monocular depth estimation,
3D spatial computing, semantic scene understanding, Open3D reconstruction,
and interactive OpenCV & Web 3D dashboards.
"""

import argparse
import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config.settings import (
    FRAME_WIDTH, FRAME_HEIGHT, WEB_HOST, WEB_PORT,
    OUTPUTS_DIR, CAMERA_INDEX
)
from core.camera import CameraStream
from core.detector import ObjectDetector
from core.depth_estimator import DepthEstimator
from core.spatial_engine import SpatialEngine
from core.scene_understanding import SemanticSceneUnderstander
from core.reconstructor import SceneReconstructor3D
from visualizer.hud_renderer import HUDRenderer
from visualizer.open3d_viewer import Open3DViewer
from visualizer.web_server import start_web_server, update_web_state


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionSpace - Real-Time 3D Scene Understanding & Spatial Computing"
    )
    parser.add_argument(
        "--source", type=str, default=str(CAMERA_INDEX),
        help="Input source: camera index (e.g. 0), video file path, image path, or 'demo'/'synthetic'"
    )
    parser.add_argument(
        "--no-gui", action="store_true",
        help="Run in headless mode without local OpenCV window (ideal for servers or background tests)"
    )
    parser.add_argument(
        "--no-web", action="store_true",
        help="Disable Flask Web 3D Visualizer"
    )
    parser.add_argument(
        "--open3d", action="store_true",
        help="Launch Open3D interactive 3D visualizer immediately on first reconstructed frame"
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Stop automatically after N frames (useful for automated testing; 0 = run indefinitely)"
    )
    parser.add_argument(
        "--host", type=str, default=WEB_HOST,
        help=f"Web server host IP (default: {WEB_HOST})"
    )
    parser.add_argument(
        "--port", type=int, default=WEB_PORT,
        help=f"Web server port (default: {WEB_PORT})"
    )
    return parser.parse_args()


def run_pipeline():
    args = parse_args()

    print("=" * 72)
    print(" VISIONSPACE: Real-Time 3D Scene Understanding Using Computer Vision")
    print(" Goel Institute of Technology & Management - CSE (AIML)")
    print("=" * 72)

    # 1. Initialize Pipeline Modules
    print("\n[1/5] Initializing Camera / Video Stream...")
    cam = CameraStream(source=args.source, width=FRAME_WIDTH, height=FRAME_HEIGHT)

    print("[2/5] Initializing Deep Learning AI Modules...")
    detector = ObjectDetector()
    depth_estimator = DepthEstimator()

    print("[3/5] Initializing Spatial Computing & Scene Understanding Engines...")
    spatial_engine = SpatialEngine()
    scene_understander = SemanticSceneUnderstander()
    reconstructor = SceneReconstructor3D()

    print("[4/5] Initializing Visualizers...")
    hud_renderer = HUDRenderer()
    o3d_viewer = Open3DViewer()

    # 2. Start Web 3D Server (if not disabled)
    if not args.no_web:
        print("[5/5] Launching Interactive Web 3D Visualizer...")
        start_web_server(host=args.host, port=args.port)
        print(f"      >> Open in browser: http://{args.host}:{args.port}")

    print("\n[VisionSpace] Perception Pipeline Active. Processing frames...\n")

    frame_idx = 0
    latest_pcd = None
    latest_spatial_objects = []

    try:
        while True:
            ret, frame = cam.read()
            if not ret or frame is None:
                print("[VisionSpace] End of stream or camera disconnected.")
                break

            frame_idx += 1

            # A. Monocular Depth Estimation (MiDaS)
            depth_meters, depth_colormap = depth_estimator.estimate(frame)

            # B. Real-Time Object Detection (YOLOv11)
            detections = detector.detect(frame)

            # C. Spatial Feature Extraction (3D Back-Projection, Centroids, Proximity)
            spatial_objects = spatial_engine.extract_spatial_features(detections, depth_meters)
            latest_spatial_objects = spatial_objects

            # D. Top-Down Bird's Eye View (BEV) Radar
            bev_radar = spatial_engine.generate_bev_radar(spatial_objects)

            # E. Semantic Scene Understanding & Situational Reasoning
            semantic_info = scene_understander.understand_scene(spatial_objects)

            # F. Calculate real-time FPS
            fps = hud_renderer.update_fps()

            # G. Draw 3D spatial annotations on RGB frame
            rgb_annotated = hud_renderer.draw_spatial_overlay(frame, spatial_objects)

            # H. Assemble full Multi-View Dashboard
            dashboard = hud_renderer.render_dashboard(
                rgb_annotated=rgb_annotated,
                depth_colormap=depth_colormap,
                bev_radar=bev_radar,
                semantic_info=semantic_info,
                fps=fps
            )

            # I. 3D Point Cloud Reconstruction (Sampled periodically or for web)
            # Reconstruct for web/open3d every 5 frames for high performance
            web_pc_payload = None
            saved_ply_name = "scene_reconstruction.ply"
            if frame_idx % 5 == 0 or frame_idx == 1:
                pts_xyz, colors_rgb = reconstructor.build_point_cloud(frame, depth_meters, stride=6)
                latest_pcd = reconstructor.create_open3d_point_cloud(pts_xyz, colors_rgb)
                web_pc_payload = reconstructor.to_web_json(pts_xyz, colors_rgb, max_points=12000)
                if frame_idx == 1:
                    reconstructor.export_point_cloud(latest_pcd, "scene_reconstruction.ply")

            # J. Update Web Dashboard State
            if not args.no_web:
                update_web_state(
                    dashboard_frame=dashboard,
                    rgb_frame=rgb_annotated,
                    depth_frame=depth_colormap,
                    raw_frame=frame,
                    depth_meters=depth_meters,
                    latest_pcd=latest_pcd,
                    spatial_objects=spatial_objects,
                    semantic_info=semantic_info,
                    web_point_cloud=web_pc_payload,
                    fps=fps,
                    ply_filename=saved_ply_name
                )

            # K. Open3D Initial Auto-Display (if requested)
            if args.open3d and frame_idx == 1 and latest_pcd is not None:
                o3d_viewer.display_scene(latest_pcd, spatial_objects)

            # L. Local OpenCV GUI Display
            if not args.no_gui:
                cv2.imshow("VisionSpace - Real-Time 3D Scene Understanding", dashboard)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q') or key == 27:  # 'q' or ESC
                    print("\n[VisionSpace] Exit requested by user.")
                    break
                elif key == ord('s'):  # Save Point Cloud (.ply)
                    pts_xyz, colors_rgb = reconstructor.build_point_cloud(frame, depth_meters, stride=3)
                    pcd = reconstructor.create_open3d_point_cloud(pts_xyz, colors_rgb)
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    fname = f"scene_{ts}.ply"
                    saved_path = reconstructor.export_point_cloud(pcd, fname)
                    update_web_state(ply_filename=fname)
                    print(f"[VisionSpace] Saved Point Cloud to: {saved_path}")
                elif key == ord('v'):  # Launch Open3D Window
                    pts_xyz, colors_rgb = reconstructor.build_point_cloud(frame, depth_meters, stride=3)
                    pcd = reconstructor.create_open3d_point_cloud(pts_xyz, colors_rgb)
                    o3d_viewer.display_scene(pcd, spatial_objects)
                elif key == ord('w'):  # Print Web URL
                    print(f"[VisionSpace] Interactive Web UI: http://{args.host}:{args.port}")

            # Optional frame limit for testing
            if 0 < args.max_frames <= frame_idx:
                print(f"[VisionSpace] Reached maximum frame count ({args.max_frames}). Completed successfully.")
                break

    except KeyboardInterrupt:
        print("\n[VisionSpace] Interrupted by user (Ctrl+C).")
    finally:
        cam.release()
        if not args.no_gui:
            cv2.destroyAllWindows()
        print("[VisionSpace] Clean shutdown complete.")


if __name__ == "__main__":
    run_pipeline()
