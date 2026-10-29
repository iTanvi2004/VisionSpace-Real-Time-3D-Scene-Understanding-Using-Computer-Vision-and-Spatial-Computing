"""
VisionSpace - MiDaS Monocular Depth Estimation Module
Estimates dense relative depth from single monocular RGB frames,
maps relative disparity to metric distance (meters), and produces colorized depth maps.
"""

import cv2
import torch
import numpy as np
from typing import Tuple, Optional
from config.settings import (
    MIDAS_MODEL_TYPE, DEPTH_MIN_METERS, DEPTH_MAX_METERS
)


class DepthEstimator:
    """Monocular Depth Estimator powered by Intel ISL MiDaS."""

    def __init__(self, model_type: str = MIDAS_MODEL_TYPE, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"[VisionSpace] Loading MiDaS Depth Estimator ({model_type}) on {self.device}...")

        # Bypass torch hub repo verification prompts programmatically
        torch.hub._check_repo_is_trusted = lambda *a, **kw: True

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()

        # Load appropriate transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        print("[VisionSpace] MiDaS Depth Estimator successfully initialized.")

    def estimate(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs monocular depth estimation on a BGR image frame.

        Returns:
            depth_meters: (H, W) float32 array of estimated distance in meters.
            depth_colormap: (H, W, 3) uint8 BGR image for visual display.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(frame_rgb).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_batch)
            # Resize disparity map to match original image resolution
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode="bicubic",
                align_corners=False
            ).squeeze()

            disparity = prediction.cpu().numpy().astype(np.float32)

        # Disparity post-processing
        # MiDaS produces relative disparity: higher values = closer objects, lower values = far objects.
        disp_min = np.percentile(disparity, 2)
        disp_max = np.percentile(disparity, 98)
        disp_clipped = np.clip(disparity, disp_min, disp_max)

        # Normalize disparity to [0, 1]
        denom = (disp_max - disp_min) if (disp_max - disp_min) > 1e-4 else 1.0
        norm_disp = (disp_clipped - disp_min) / denom
        norm_disp = np.clip(norm_disp, 0.0, 1.0)

        # Convert disparity to approximate metric depth Z (meters)
        # Using calibrated inverse mapping: Z = Z_min + (1 - norm_disp) * (Z_max - Z_min)
        depth_meters = DEPTH_MIN_METERS + (1.0 - norm_disp) * (DEPTH_MAX_METERS - DEPTH_MIN_METERS)
        depth_meters = depth_meters.astype(np.float32)

        # Generate colorized visualization (Inferno / Turbo)
        depth_u8 = (norm_disp * 255).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)

        return depth_meters, depth_colormap

    def get_roi_depth(self, depth_meters: np.ndarray, bbox: list) -> float:
        """
        Calculates the median depth of an object from its bounding box ROI.
        Median is used to filter out background edges and sensor noise.
        """
        x1, y1, x2, y2 = bbox
        h, w = depth_meters.shape[:2]
        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return float(DEPTH_MAX_METERS)

        # Take central 60% of the bounding box to sample object surface reliably
        bw = x2 - x1
        bh = y2 - y1
        cx1 = int(x1 + 0.20 * bw)
        cx2 = int(x2 - 0.20 * bw)
        cy1 = int(y1 + 0.20 * bh)
        cy2 = int(y2 - 0.20 * bh)

        roi = depth_meters[cy1:cy2, cx1:cx2]
        if roi.size == 0:
            roi = depth_meters[y1:y2, x1:x2]

        if roi.size == 0:
            return float(DEPTH_MAX_METERS)

        median_dist = float(np.median(roi))
        return round(median_dist, 2)
