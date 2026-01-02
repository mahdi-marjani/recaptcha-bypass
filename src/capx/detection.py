import math
import cv2
import numpy as np
from PIL import Image

from .config import YOLO_MODELS, IMAGES_DIRECTORY
from .utils import (
    find_object_locations,
    locations_to_numbers,
    find_filled_cells,
)

# =========================
# Basic helpers
# =========================

def _load_main_image(timestamp: str) -> np.ndarray:
    image = Image.open(IMAGES_DIRECTORY / f"0-{timestamp}.png")
    return np.asarray(image)


def _run_model_for_target(image, target_num):
    if target_num == 1001:
        return YOLO_MODELS["crosswalk"].predict(image, conf=0.4), 0
    if target_num == 1002:
        return YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4), 489
    if target_num == 1003:
        return YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4), 522

    return YOLO_MODELS["yolo11x"].predict(image, conf=0.4), target_num


def _find_target_boxes(result, target_num):
    return [
        i for i, cls in enumerate(result[0].boxes.cls)
        if cls == target_num
    ]


# =========================
# 3x3 captcha
# =========================

def detect_cells_3x3(target_num, timestamp):
    image = _load_main_image(timestamp)
    result, target_num = _run_model_for_target(image, target_num)

    target_boxes = _find_target_boxes(result, target_num)
    boxes = result[0].boxes.data

    cells = set()

    for idx in target_boxes:
        box = boxes[idx]
        xc = (box[0] + box[2]) / 2
        yc = (box[1] + box[3]) / 2

        col = int(xc / (image.shape[1] / 3))
        row = int(yc / (image.shape[0] / 3))

        cells.add(row * 3 + col + 1)

    return list(cells)


# =========================
# 4x4 captcha
# =========================

def detect_cells_4x4(target_num, timestamp):
    image = _load_main_image(timestamp)

    if target_num < 1000:
        return _detect_4x4_with_seg(image, target_num)

    return _detect_4x4_with_boxes(image, target_num)


# =========================
# Segmentation based
# =========================

def _detect_4x4_with_seg(image, target_num):
    result = YOLO_MODELS["yolo11x-seg"].predict(image, conf=0.4)
    target_boxes = _find_target_boxes(result, target_num)

    cells = []

    for idx in target_boxes:
        if result[0].masks is None:
            continue

        mask = result[0].masks[idx].cpu().data.numpy().transpose(1, 2, 0)
        mask = cv2.merge((mask, mask, mask))

        h, w, _ = result[0].orig_img.shape
        mask = cv2.resize(mask, (w, h))
        mask = _make_binary_mask(mask)

        masked = cv2.bitwise_and(
            result[0].orig_img,
            result[0].orig_img,
            mask=mask
        )

        locations = find_object_locations(masked, rows=4, cols=4, min_pixels=100)
        cells.extend(locations_to_numbers(locations))

    return sorted(set(cells))


def _make_binary_mask(mask):
    hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(mask, (0, 0, 0), (0, 0, 1))
    return cv2.bitwise_not(mask)


# =========================
# Box based
# =========================

def _detect_4x4_with_boxes(image, target_num):
    result, target_num = _run_model_for_target(image, target_num)
    boxes = result[0].boxes.data

    target_boxes = _find_target_boxes(result, target_num)
    cells = []

    for idx in target_boxes:
        cells.extend(_box_to_4x4_cells(image, boxes[idx]))

    return sorted(set(cells))


def _box_to_4x4_cells(image, box):
    x1, y1, x4, y4 = map(int, box[:4])

    corners = [
        (x1, y1),
        (x4, y1),
        (x1, y4),
        (x4, y4),
    ]

    cell_size = image.shape[0] / 4
    max_x = max(p[0] for p in corners)
    max_y = max(p[1] for p in corners)

    cells = []

    for x, y in corners:
        row = math.floor(y / cell_size) + 1
        col = math.floor(x / cell_size) + 1

        if math.isclose(y % cell_size, 0) and math.isclose(y, max_y):
            row -= 1
        if math.isclose(x % cell_size, 0) and math.isclose(x, max_x):
            col -= 1

        row = max(1, min(4, row))
        col = max(1, min(4, col))

        cells.append((row - 1) * 4 + col)

    return find_filled_cells(cells)
