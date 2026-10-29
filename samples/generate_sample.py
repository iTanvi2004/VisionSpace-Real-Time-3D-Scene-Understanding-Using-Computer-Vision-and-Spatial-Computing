"""
VisionSpace - Sample Image Generator
Generates realistic benchmark test images for testing the perception pipeline.
"""

import cv2
import numpy as np
from pathlib import Path
from config.settings import SAMPLES_DIR


def generate_benchmark_room_image(output_path: Path = SAMPLES_DIR / "sample_room.jpg") -> str:
    """
    Renders a realistic synthetic indoor room scene with:
    - Office desk with wood grain
    - Laptop computer
    - Ceramic coffee cup
    - Chair
    - Potted plant
    - Wall poster / window
    """
    w, h = 640, 480
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # 1. Background Wall (Subtle vertical lighting gradient)
    for y in range(int(h * 0.62)):
        val = int(225 - (y / (h * 0.62)) * 35)
        canvas[y, :] = [val - 12, val, val + 8]

    # 2. Hardwood Floor
    for y in range(int(h * 0.62), h):
        prog = (y - h * 0.62) / (h * 0.38)
        canvas[y, :] = [int(45 + 25 * prog), int(80 + 40 * prog), int(135 + 45 * prog)]

    # 3. Wall Window / Art Frame
    cv2.rectangle(canvas, (50, 40), (200, 180), (250, 248, 240), -1)
    cv2.rectangle(canvas, (50, 40), (200, 180), (80, 70, 60), 4)
    cv2.line(canvas, (125, 40), (125, 180), (80, 70, 60), 3)
    cv2.line(canvas, (50, 110), (200, 110), (80, 70, 60), 3)

    # 4. Large Desk Tabletop (Perspective trapezoid)
    desk_y = int(h * 0.58)
    table_pts = np.array([
        [100, desk_y],
        [540, desk_y],
        [600, h - 15],
        [40, h - 15]
    ], np.int32)
    cv2.fillPoly(canvas, [table_pts], (185, 155, 125))
    cv2.polylines(canvas, [table_pts], True, (120, 90, 60), 3)

    # 5. Ergonomic Office Chair
    chair_cx = 320
    cv2.rectangle(canvas, (chair_cx - 55, desk_y - 130), (chair_cx + 55, desk_y - 10), (60, 65, 120), -1)
    cv2.rectangle(canvas, (chair_cx - 48, desk_y - 123), (chair_cx + 48, desk_y - 17), (75, 80, 145), -1)
    # Headrest
    cv2.rectangle(canvas, (chair_cx - 30, desk_y - 170), (chair_cx + 30, desk_y - 135), (55, 60, 110), -1)

    # 6. Laptop Computer
    laptop_cx = 320
    # Screen bezel
    cv2.rectangle(canvas, (laptop_cx - 80, desk_y - 95), (laptop_cx + 80, desk_y), (35, 35, 40), -1)
    # Display screen with colorful browser mockup
    cv2.rectangle(canvas, (laptop_cx - 74, desk_y - 89), (laptop_cx + 74, desk_y - 6), (225, 160, 30), -1)
    cv2.rectangle(canvas, (laptop_cx - 70, desk_y - 45), (laptop_cx + 70, desk_y - 10), (245, 245, 250), -1)
    # Keyboard base
    kb_pts = np.array([
        [laptop_cx - 80, desk_y],
        [laptop_cx + 80, desk_y],
        [laptop_cx + 105, desk_y + 45],
        [laptop_cx - 105, desk_y + 45]
    ], np.int32)
    cv2.fillPoly(canvas, [kb_pts], (80, 80, 85))
    cv2.polylines(canvas, [kb_pts], True, (40, 40, 45), 2)
    # Trackpad
    cv2.rectangle(canvas, (laptop_cx - 25, desk_y + 25), (laptop_cx + 25, desk_y + 40), (110, 110, 115), -1)

    # 7. Coffee Mug
    cup_x = laptop_cx + 140
    cup_y = desk_y + 25
    cv2.rectangle(canvas, (cup_x - 18, cup_y - 30), (cup_x + 18, cup_y + 12), (245, 245, 245), -1)
    cv2.ellipse(canvas, (cup_x, cup_y + 12), (18, 7), 0, 0, 360, (215, 215, 215), -1)
    cv2.ellipse(canvas, (cup_x, cup_y - 30), (18, 7), 0, 0, 360, (90, 70, 50), -1)
    cv2.ellipse(canvas, (cup_x + 19, cup_y - 9), (8, 14), 0, 270, 90, (230, 230, 230), 3)

    # 8. Potted Desk Plant
    plant_x = 115
    plant_y = desk_y - 5
    # Clay pot
    pot_pts = np.array([
        [plant_x - 22, plant_y],
        [plant_x + 22, plant_y],
        [plant_x + 15, plant_y + 45],
        [plant_x - 15, plant_y + 45]
    ], np.int32)
    cv2.fillPoly(canvas, [pot_pts], (65, 85, 195))
    # Green Foliage
    cv2.circle(canvas, (plant_x - 14, plant_y - 25), 22, (60, 155, 55), -1)
    cv2.circle(canvas, (plant_x + 15, plant_y - 28), 20, (50, 135, 45), -1)
    cv2.circle(canvas, (plant_x, plant_y - 40), 22, (75, 175, 65), -1)
    cv2.circle(canvas, (plant_x - 2, plant_y - 18), 16, (40, 120, 35), -1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), canvas)
    print(f"[VisionSpace] Generated benchmark room image: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    generate_benchmark_room_image()
