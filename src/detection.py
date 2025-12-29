import cv2
import random
import numpy as np
from math import ceil
from PIL import Image
from datetime import datetime
from .config import YOLO_MODELS, IMAGES_DIRECTORY
from .utils import detect_car_locations, convert_to_position_indices, get_occupied_cells

def get_answers(target_num, timestamp):
    image = Image.open(IMAGES_DIRECTORY.joinpath(f"0-{timestamp}.png"))
    image = np.asarray(image)

    if target_num == 1001:
        result = YOLO_MODELS["crosswalk"].predict(image)
        target_num = 0
    elif target_num == 1002:
        result = YOLO_MODELS["yolov8x-oiv7"].predict(image)
        target_num = 489
    elif target_num == 1003:
        result = YOLO_MODELS["yolov8x-oiv7"].predict(image)
        target_num = 522
    else:
        result = YOLO_MODELS["yolo11x"].predict(image)

    target_index = [i for i, num in enumerate(result[0].boxes.cls) if num == target_num]

    answers = set()

    boxes = result[0].boxes.data
    for i in target_index:
        target_box = boxes[i]
        xc, yc = (target_box[0] + target_box[2]) / 2, (
            target_box[1] + target_box[3]
        ) / 2

        x_pos = int(xc / (image.shape[1] / 3))
        y_pos = int(yc / (image.shape[0] / 3))

        point_num = y_pos * 3 + x_pos + 1

        answers.add(point_num)

    return list(answers)

def get_answers_4(target_num, timestamp):
    image = Image.open(IMAGES_DIRECTORY.joinpath(f"0-{timestamp}.png"))
    image = np.asarray(image)

    if target_num < 1000:
        result_seg = YOLO_MODELS["yolo11x-seg"].predict(image)

        target_index = []
        count = 0
        for num in result_seg[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1

        answers = []

        for i in target_index:
            if result_seg[0].masks is not None:
                mask_raw = result_seg[0].masks[i].cpu().data.numpy().transpose(1, 2, 0)
                mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
                h2, w2, c2 = result_seg[0].orig_img.shape
                mask = cv2.resize(mask_3channel, (w2, h2))
                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([0, 0, 1])
                mask = cv2.inRange(mask, lower_black, upper_black)
                mask = cv2.bitwise_not(mask)
                masked = cv2.bitwise_and(
                    result_seg[0].orig_img, result_seg[0].orig_img, mask=mask
                )
                car_locations = detect_car_locations(
                    masked, num_rows=4, num_columns=4, threshold=100
                )
                position_indices = convert_to_position_indices(car_locations)
                for indice in position_indices:
                    answers.append(indice)

        answers = sorted(list(answers))
        return list(set(answers))
    else:
        if target_num == 1001:
            result = YOLO_MODELS["crosswalk"].predict(image)
            target_num = 0
        elif target_num == 1002:
            result = YOLO_MODELS["yolov8x-oiv7"].predict(image)
            target_num = 489
        elif target_num == 1003:
            result = YOLO_MODELS["yolov8x-oiv7"].predict(image)
            target_num = 522

        boxes = result[0].boxes.data

        target_index = []
        count = 0
        for num in result[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1

        answers = []
        for i in target_index:
            target_box = boxes[i]
            p1, p2 = (int(target_box[0]), int(target_box[1])), (
                int(target_box[2]),
                int(target_box[3]),
            )
            x1, y1 = p1
            x4, y4 = p2
            x2 = x4
            y2 = y1
            x3 = x1
            y3 = y4
            xys = [x1, y1, x2, y2, x3, y3, x4, y4]

            four_cells = []
            for i in target_index:
                target_box = boxes[i]
                p1, p2 = (int(target_box[0]), int(target_box[1])), (
                    int(target_box[2]),
                    int(target_box[3]),
                )
                x1, y1 = p1
                x4, y4 = p2

                x2 = x4
                y2 = y1
                x3 = x1
                y3 = y4

                four_cells = []

                for i in range(4):
                    x = xys[i * 2]
                    y = xys[(i * 2) + 1]
                    cell_size = image.shape[1] / 4
                    row = ceil(y / cell_size)
                    column = ceil(x / cell_size)
                    cell_number = (row - 1) * 4 + column
                    four_cells.append(cell_number)

                answer = get_occupied_cells(four_cells)
                for ans in answer:
                    answers.append(ans)
        answers = sorted(list(answers))
        return list(set(answers))