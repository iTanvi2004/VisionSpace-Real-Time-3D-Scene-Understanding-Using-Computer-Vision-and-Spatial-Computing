"""
VisionSpace - Camera and Video Stream Module
Supports live Monocular/RGB webcams, pre-recorded video files, still images,
and a synthetic procedural test stream for automated testing without hardware camera.
"""

import cv2
import numpy as np
import time
from typing import Optional, Tuple, Generator
from config.settings import FRAME_WIDTH, FRAME_HEIGHT, CAMERA_INDEX


class CameraStream:
    """Unified camera and video feed manager."""

    def __init__(self, source: Optional[str or int] = None, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        self.width = width
        self.height = height
        self.source = CAMERA_INDEX if source is None else source
        self.is_synthetic = False
        self.cap = None
        self._single_image = None
        self._frame_count = 0
        self._init_source()

    def _init_source(self):
        """Initialize the video capture or synthetic source."""
        if isinstance(self.source, str):
            # Check if source is a test/demo keyword
            if self.source.lower() in ("synthetic", "demo", "simulated", "mock"):
                self.is_synthetic = True
                return

            # Check if source is a still image
            if self.source.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
                img = cv2.imread(self.source)
                if img is not None:
                    self._single_image = cv2.resize(img, (self.width, self.height))
                    return
                else:
                    print(f"[VisionSpace Warning] Image '{self.source}' not found. Falling back to synthetic generator.")
                    self.is_synthetic = True
                    return

            # Try video file or numeric string
            if self.source.isdigit():
                self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(self.source)
        else:
            # Numeric camera index with DirectShow backend for faster startup on Windows
            self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)

        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                print(f"[VisionSpace] Camera device {self.source} could not be opened. Using synthetic test stream.")
                self.is_synthetic = True

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame from the stream."""
        if self.is_synthetic:
            self._frame_count += 1
            return True, self._generate_synthetic_frame()

        if self._single_image is not None:
            self._frame_count += 1
            # Add slight time-based movement or return clean frame
            return True, self._single_image.copy()

        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret and frame is not None:
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height))
                self._frame_count += 1
                return True, frame
            else:
                # Video file reached end -> rewind if video file
                if isinstance(self.source, str) and not self.source.isdigit():
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        frame = cv2.resize(frame, (self.width, self.height))
                        return True, frame
                return False, None

        return False, None

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Generates a rich, realistic synthetic room scene for testing.
        Contains a desk, laptop, coffee cup, chair, and wall poster.
        """
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        # Background wall gradient (cream/soft gray)
        for y in range(int(self.height * 0.65)):
            val = int(220 - (y / (self.height * 0.65)) * 30)
            canvas[y, :] = [val - 10, val, val + 10]

        # Floor (warm wood tone gradient)
        for y in range(int(self.height * 0.65), self.height):
            prog = (y - self.height * 0.65) / (self.height * 0.35)
            canvas[y, :] = [int(40 + 20 * prog), int(70 + 35 * prog), int(120 + 40 * prog)]

        # Wall Window / Poster
        cv2.rectangle(canvas, (40, 40), (180, 160), (250, 245, 230), -1)
        cv2.rectangle(canvas, (40, 40), (180, 160), (100, 90, 80), 3)
        cv2.line(canvas, (110, 40), (110, 160), (100, 90, 80), 2)
        cv2.line(canvas, (40, 100), (180, 100), (100, 90, 80), 2)

        # Desk (table top)
        desk_top_y = int(self.height * 0.58)
        table_pts = np.array([
            [120, desk_top_y],
            [520, desk_top_y],
            [580, self.height - 20],
            [60, self.height - 20]
        ], np.int32)
        cv2.fillPoly(canvas, [table_pts], (180, 150, 120))
        cv2.polylines(canvas, [table_pts], True, (130, 100, 70), 3)

        # Laptop on table (Screen + Keyboard base)
        laptop_cx = 320
        # Gentle dynamic oscillation so real-time motion detection can be observed
        shift_x = int(5 * np.sin(self._frame_count * 0.08))
        laptop_cx += shift_x

        # Laptop screen
        cv2.rectangle(canvas, (laptop_cx - 70, desk_top_y - 80), (laptop_cx + 70, desk_top_y), (40, 40, 40), -1)
        cv2.rectangle(canvas, (laptop_cx - 65, desk_top_y - 75), (laptop_cx + 65, desk_top_y - 5), (220, 160, 40), -1)
        # Laptop keyboard base
        base_pts = np.array([
            [laptop_cx - 70, desk_top_y],
            [laptop_cx + 70, desk_top_y],
            [laptop_cx + 90, desk_top_y + 40],
            [laptop_cx - 90, desk_top_y + 40]
        ], np.int32)
        cv2.fillPoly(canvas, [base_pts], (70, 70, 70))
        cv2.polylines(canvas, [base_pts], True, (30, 30, 30), 2)

        # Coffee Cup on table
        cup_x = laptop_cx + 120
        cup_y = desk_top_y + 20
        cv2.rectangle(canvas, (cup_x - 15, cup_y - 25), (cup_x + 15, cup_y + 10), (240, 240, 240), -1)
        cv2.ellipse(canvas, (cup_x, cup_y + 10), (15, 6), 0, 0, 360, (210, 210, 210), -1)
        cv2.ellipse(canvas, (cup_x, cup_y - 25), (15, 6), 0, 0, 360, (100, 80, 60), -1)
        # Cup handle
        cv2.ellipse(canvas, (cup_x + 16, cup_y - 8), (7, 12), 0, 270, 90, (230, 230, 230), 3)

        # Chair backrest behind desk
        chair_x = laptop_cx
        cv2.rectangle(canvas, (chair_x - 50, desk_top_y - 120), (chair_x + 50, desk_top_y - 15), (60, 60, 140), -1)
        cv2.rectangle(canvas, (chair_x - 45, desk_top_y - 115), (chair_x + 45, desk_top_y - 20), (70, 70, 170), -1)

        # Potted Plant on left table corner
        plant_x = 130
        plant_y = desk_top_y - 10
        # Pot
        pot_pts = np.array([
            [plant_x - 18, plant_y],
            [plant_x + 18, plant_y],
            [plant_x + 12, plant_y + 35],
            [plant_x - 12, plant_y + 35]
        ], np.int32)
        cv2.fillPoly(canvas, [pot_pts], (60, 80, 190))
        # Leaves
        cv2.circle(canvas, (plant_x - 12, plant_y - 20), 18, (60, 160, 50), -1)
        cv2.circle(canvas, (plant_x + 12, plant_y - 22), 16, (50, 140, 40), -1)
        cv2.circle(canvas, (plant_x, plant_y - 32), 17, (70, 180, 60), -1)

        # Timestamp watermark
        ts = time.strftime("%H:%M:%S")
        cv2.putText(canvas, f"SYNTHETIC DEMO FEED | {ts}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        cv2.putText(canvas, f"SYNTHETIC DEMO FEED | {ts}", (15, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1)

        return canvas

    def release(self):
        """Release video capture device."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
