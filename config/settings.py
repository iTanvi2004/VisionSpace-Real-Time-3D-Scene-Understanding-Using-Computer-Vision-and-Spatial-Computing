"""
VisionSpace - Configuration and Settings
Centralized parameters for camera, deep learning models, spatial computing, and visualization.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLES_DIR = BASE_DIR / "samples"
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)

# Device Configuration
# Uses CUDA if available, falls back gracefully to CPU
DEVICE = "cpu"  # Will be auto-checked at runtime

# Camera / Video Stream Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
TARGET_FPS = 30

# Camera Intrinsics (Pinhole Camera Model Approximation for 640x480 standard webcam)
# Focal length (fx, fy) typically ~0.8 * image_width for 60-70 deg FOV webcams
FOCAL_LENGTH_X = 500.0  # pixels
FOCAL_LENGTH_Y = 500.0  # pixels
PRINCIPAL_POINT_X = FRAME_WIDTH / 2.0   # cx = 320.0
PRINCIPAL_POINT_Y = FRAME_HEIGHT / 2.0  # cy = 240.0

# YOLOv11 Object Detection Configuration
YOLO_MODEL_NAME = "yolo11n.pt"  # Ultra-fast real-time nano model
DETECTION_CONFIDENCE = 0.40
DETECTION_IOU_THRESHOLD = 0.45
# Focus classes for spatial awareness (indoor + outdoor relevant)
INTEREST_CLASSES = [
    "person", "chair", "couch", "potted plant", "bed", "dining table",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "bottle", "cup", "fork", "knife", "spoon", "bowl", "book", "clock",
    "bicycle", "car", "motorcycle", "bus", "truck", "traffic light", "fire hydrant",
    "stop sign", "backpack", "umbrella", "handbag", "suitcase", "sports ball"
]

# MiDaS Monocular Depth Estimation Configuration
MIDAS_MODEL_TYPE = "MiDaS_small"  # MiDaS_small is optimized for real-time inference
DEPTH_INPUT_SIZE = 256
DEPTH_SCALE_FACTOR = 1.0          # Multiplier to scale relative inverse depth
DEPTH_MIN_METERS = 0.3            # Minimum estimated distance (30 cm)
DEPTH_MAX_METERS = 10.0           # Maximum estimated distance (10 meters)

# Spatial Distance Alert Thresholds (Meters)
DISTANCE_THRESHOLD_CRITICAL = 0.9   # Red hazard zone (< 0.9m)
DISTANCE_THRESHOLD_WARNING = 1.8    # Yellow caution zone (0.9m - 1.8m)
                                    # Green safe zone (> 1.8m)

# 3D Point Cloud Settings
POINT_CLOUD_VOXEL_SIZE = 0.04       # Voxel downsampling leaf size (4 cm)
POINT_CLOUD_MAX_POINTS = 50000      # Cap points for high-performance rendering
OUTLIER_NB_NEIGHBORS = 20           # Statistical outlier removal neighbors
OUTLIER_STD_RATIO = 2.0             # Statistical outlier std ratio

# Web Server Settings
WEB_HOST = "127.0.0.1"
WEB_PORT = 5000
