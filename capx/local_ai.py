from capx_core.detector import detect_cells

AVAILABLE_MODELS = [
    "bicycle",
    "bus",
    "tractor",
    "boat",
    "car",
    "hydrant",
    "motorcycle",
    "traffic",
    "crosswalk",
    "stair",
    "taxi",
]

def is_model_available(target_text):
    for model in AVAILABLE_MODELS:
        if model in target_text:
            return True
    return False

def detect(image_array, grid, target_text):
    """
    Detect cells in the image based on the grid and target text.
    
    :param image_array: Numpy array of the image.
    :param grid: Grid size, either "3x3" or "4x4".
    :param target_text: Text to detect or target in the image.
    :return: List of detected cells from the capx core.
    """
    
    cells = detect_cells(
        image=image_array,
        grid=grid,
        target_text=target_text
    )

    return cells