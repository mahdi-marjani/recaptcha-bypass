import cv2
import numpy as np
import math
from PIL import Image
from .config import YOLO_MODELS, IMAGES_DIRECTORY
from .utils import find_object_locations, locations_to_numbers, find_filled_cells

def get_answers(target_num, timestamp):
    image = Image.open(IMAGES_DIRECTORY.joinpath(f"0-{timestamp}.png"))
    image = np.asarray(image)

    if target_num == 1001:
        result = YOLO_MODELS["crosswalk"].predict(image, conf=0.4)
        target_num = 0
    elif target_num == 1002:
        result = YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4)
        target_num = 489
    elif target_num == 1003:
        result = YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4)
        target_num = 522
    else:
        result = YOLO_MODELS["yolo11x"].predict(image, conf=0.4)

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
        result_seg = YOLO_MODELS["yolo11x-seg"].predict(image, conf=0.4)

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
                car_locations = find_object_locations(
                    masked, rows=4, cols=4, min_pixels=100
                )
                position_indices = locations_to_numbers(car_locations)
                for indice in position_indices:
                    answers.append(indice)

        answers = sorted(list(answers))
        return list(set(answers))
    else:
        if target_num == 1001:
            result = YOLO_MODELS["crosswalk"].predict(image, conf=0.4)
            target_num = 0
        elif target_num == 1002:
            result = YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4)
            target_num = 489
        elif target_num == 1003:
            result = YOLO_MODELS["yolov8x-oiv7"].predict(image, conf=0.4)
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

            width = image.shape[0]
            cell_size = width / 4.0

            points = [(xys[2*i], xys[2*i+1]) for i in range(4)]
            max_x = max(p[0] for p in points)
            max_y = max(p[1] for p in points)

            four_cells = []
            for i in range(4):
                x, y = points[i]
                
                is_right = math.isclose(x, max_x)
                is_bottom = math.isclose(y, max_y)
                
                row = math.floor(y / cell_size) + 1
                col = math.floor(x / cell_size) + 1
                
                if math.isclose(y % cell_size, 0):
                    if is_bottom:
                        row -= 1
                
                if math.isclose(x % cell_size, 0):
                    if is_right:
                        col -= 1
                
                row = max(1, min(4, row))
                col = max(1, min(4, col))
                
                cell_number = (row - 1) * 4 + col
                four_cells.append(cell_number)

            answer = find_filled_cells(four_cells)
            for ans in answer:
                answers.append(ans)
        answers = sorted(list(answers))
        return list(set(answers))