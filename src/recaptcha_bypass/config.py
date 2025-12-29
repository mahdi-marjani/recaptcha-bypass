import re
from pathlib import Path
from ultralytics import YOLO

CURRENT_DIRECTORY = Path().absolute()
MODELS_DIRECTORY = CURRENT_DIRECTORY / "models"
IMAGES_DIRECTORY = CURRENT_DIRECTORY / "images"

YOLO_MODELS = {
    "yolo11x": YOLO(MODELS_DIRECTORY / "yolo11x.pt"),
    "crosswalk": YOLO(MODELS_DIRECTORY / "crosswalk.pt"),
    "yolo11x-seg": YOLO(MODELS_DIRECTORY / "yolo11x-seg.pt"),
    "yolov8x-oiv7": YOLO(MODELS_DIRECTORY / "yolov8x-oiv7.pt"),
}

TARGET_MAPPING = {
    "bicycle": 1,
    "bus": 5,
    "tractor": 7,
    "boat": 8,
    "car": 2,
    "hydrant": 10,
    "motorcycle": 3,
    "traffic": 9,
    "crosswalk": 1001,
    "stair": 1002,
    "taxi": 1003,
}

def get_target_num(target_text):
    for key, value in TARGET_MAPPING.items():
        if re.search(key, target_text) is not None:
            return value
    return 1000

# Try loading an image with a model
try:
    from PIL import Image
    import numpy as np
    image = Image.open(IMAGES_DIRECTORY / "0.png")
    image = np.asarray(image)
    YOLO_MODELS["yolo11x"].predict(image)
except:
    pass