import re
import numpy as np
from PIL import Image
from pathlib import Path
from ultralytics import YOLO
from .model_downloader import download_model_if_missing

CURRENT_DIRECTORY = Path().absolute()

MODELS_DIRECTORY = CURRENT_DIRECTORY / "models"
MODELS_DIRECTORY.mkdir(exist_ok=True)

IMAGES_DIRECTORY = CURRENT_DIRECTORY / "images"
IMAGES_DIRECTORY.mkdir(exist_ok=True)

download_model_if_missing("crosswalk.pt", MODELS_DIRECTORY)

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


test_image = Image.new("RGB", (300, 300), color="white")
test_image = np.asarray(test_image)
YOLO_MODELS["yolo11x"].predict(test_image)