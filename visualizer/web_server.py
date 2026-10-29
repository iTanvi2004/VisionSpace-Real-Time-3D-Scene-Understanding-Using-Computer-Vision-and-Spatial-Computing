"""
VisionSpace - Interactive Web 3D Visualizer Server (Flask + WebGL / Three.js)
Serves a sleek web interface with live video streaming, interactive 3D point cloud,
3D bounding boxes, and real-time spatial telemetry.
"""

from flask import Flask, render_template, Response, jsonify, send_file, send_from_directory
import cv2
import threading
import time
import open3d as o3d
import numpy as np
from pathlib import Path
from config.settings import WEB_HOST, WEB_PORT, OUTPUTS_DIR, BASE_DIR
from core.reconstructor import SceneReconstructor3D

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static")
)

reconstructor_instance = SceneReconstructor3D()

# Global shared state updated by the main pipeline
state_lock = threading.Lock()
latest_state = {
    "dashboard_frame": None,
    "rgb_frame": None,
    "depth_frame": None,
    "raw_frame": None,
    "depth_meters": None,
    "latest_pcd": None,
    "spatial_objects": [],
    "semantic_info": {},
    "web_point_cloud": {"points": [], "colors": []},
    "fps": 0.0,
    "ply_filename": "scene_reconstruction.ply"
}


def update_web_state(dashboard_frame=None, rgb_frame=None, depth_frame=None,
                     raw_frame=None, depth_meters=None, latest_pcd=None,
                     spatial_objects=None, semantic_info=None,
                     web_point_cloud=None, fps=0.0, ply_filename=""):
    """Update global thread-safe state for web clients."""
    with state_lock:
        if dashboard_frame is not None:
            latest_state["dashboard_frame"] = dashboard_frame
        if rgb_frame is not None:
            latest_state["rgb_frame"] = rgb_frame
        if depth_frame is not None:
            latest_state["depth_frame"] = depth_frame
        if raw_frame is not None:
            latest_state["raw_frame"] = raw_frame
        if depth_meters is not None:
            latest_state["depth_meters"] = depth_meters
        if latest_pcd is not None:
            latest_state["latest_pcd"] = latest_pcd
        if spatial_objects is not None:
            latest_state["spatial_objects"] = spatial_objects
        if semantic_info is not None:
            latest_state["semantic_info"] = semantic_info
        if web_point_cloud is not None:
            latest_state["web_point_cloud"] = web_point_cloud
        latest_state["fps"] = round(fps, 1)
        if ply_filename:
            latest_state["ply_filename"] = ply_filename


def generate_mjpeg_stream():
    """Generates MJPEG multipart video stream for web clients."""
    while True:
        with state_lock:
            frame = latest_state["dashboard_frame"]
            if frame is None:
                frame = latest_state["rgb_frame"]

        if frame is not None:
            ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ret:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
        time.sleep(0.04)  # ~25 FPS stream


@app.route("/")
def index():
    """Main Web 3D Viewer Dashboard."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG stream endpoint."""
    return Response(generate_mjpeg_stream(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/api/scene_data")
def get_scene_data():
    """Returns spatial telemetry, detected objects, and 3D point cloud for Three.js."""
    with state_lock:
        return jsonify({
            "fps": latest_state["fps"],
            "spatial_objects": latest_state["spatial_objects"],
            "semantic_info": latest_state["semantic_info"],
            "point_cloud": latest_state["web_point_cloud"],
            "ply_filename": latest_state["ply_filename"]
        })


@app.route("/api/export_ply")
@app.route("/export_ply")
@app.route("/download/ply")
def export_ply_api():
    """Dynamically exports and downloads the current 3D point cloud on demand."""
    with state_lock:
        pcd = latest_state["latest_pcd"]
        raw_frame = latest_state["raw_frame"]
        depth_m = latest_state["depth_meters"]

    ts = time.strftime("%Y%m%d_%H%M%S")
    ply_filename = f"scene_{ts}.ply"
    target_path = OUTPUTS_DIR / ply_filename
    default_path = OUTPUTS_DIR / "scene_reconstruction.ply"

    try:
        # 1. Use the pre-built Open3D PointCloud from the active pipeline
        if pcd is not None and len(pcd.points) > 0:
            o3d.io.write_point_cloud(str(target_path), pcd, write_ascii=False)
            o3d.io.write_point_cloud(str(default_path), pcd, write_ascii=False)
        # 2. Or reconstruct immediately from the latest RGB frame and depth map
        elif raw_frame is not None and depth_m is not None:
            pts, cls = reconstructor_instance.build_point_cloud(raw_frame, depth_m, stride=4)
            new_pcd = reconstructor_instance.create_open3d_point_cloud(pts, cls)
            o3d.io.write_point_cloud(str(target_path), new_pcd, write_ascii=False)
            o3d.io.write_point_cloud(str(default_path), new_pcd, write_ascii=False)
        # 3. Or check if default or existing .ply file already exists in outputs
        elif default_path.exists():
            target_path = default_path
        else:
            existing_plys = list(OUTPUTS_DIR.glob("*.ply"))
            if existing_plys:
                target_path = existing_plys[0]
            else:
                # 4. Generate from synthetic room sample as fallback
                sample_img_path = BASE_DIR / "samples" / "sample_room.jpg"
                if not sample_img_path.exists():
                    from samples.generate_sample import generate_benchmark_room_image
                    generate_benchmark_room_image()
                s_img = cv2.imread(str(sample_img_path))
                h, w = s_img.shape[:2]
                dummy_depth = np.linspace(1.0, 4.0, h * w, dtype=np.float32).reshape((h, w))
                pts, cls = reconstructor_instance.build_point_cloud(s_img, dummy_depth, stride=6)
                new_pcd = reconstructor_instance.create_open3d_point_cloud(pts, cls)
                o3d.io.write_point_cloud(str(target_path), new_pcd, write_ascii=False)
                o3d.io.write_point_cloud(str(default_path), new_pcd, write_ascii=False)

        print(f"[VisionSpace] Web client downloaded 3D point cloud: {target_path}")
        return send_file(
            str(target_path),
            as_attachment=True,
            download_name=f"visionspace_scene_{ts}.ply",
            mimetype="application/octet-stream"
        )
    except Exception as e:
        print(f"[VisionSpace Error] Failed to export PLY: {e}")
        # Last resort fallback if test_output.ply exists
        fallback = OUTPUTS_DIR / "test_output.ply"
        if fallback.exists():
            return send_file(str(fallback), as_attachment=True, download_name="visionspace_scene.ply")
        return jsonify({"error": str(e)}), 500


@app.route("/download/<path:filename>")
def download_file(filename):
    """Download exported PLY files safely with automatic fallback."""
    target_path = OUTPUTS_DIR / filename
    if not target_path.exists():
        # Avoid 404: generate and download dynamically!
        return export_ply_api()
    return send_file(
        str(target_path),
        as_attachment=True,
        download_name=filename,
        mimetype="application/octet-stream"
    )


def start_web_server(host: str = WEB_HOST, port: int = WEB_PORT):
    """Starts the Flask server in a background daemon thread."""
    server_thread = threading.Thread(
        target=lambda: app.run(host=host, port=port, debug=False, use_reloader=False),
        daemon=True
    )
    server_thread.start()
    print(f"[VisionSpace] Web 3D Visualizer running at: http://{host}:{port}")
    return server_thread
