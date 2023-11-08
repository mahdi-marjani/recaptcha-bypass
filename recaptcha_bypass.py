import re
import cv2
import shutil
import random
import requests
import numpy as np
from PIL import Image
from math import ceil
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime
from os import listdir, remove
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

current_directory = Path().absolute()
models_directory = current_directory.joinpath("models")
images_directory = current_directory.joinpath("images")

# Define YOLO models
yolo_models = {
    "yolov8x": YOLO(models_directory.joinpath("yolov8x.pt")),
    "crosswalk": YOLO(models_directory.joinpath("crosswalk.pt")),
    "yolov8x-seg": YOLO(models_directory.joinpath("yolov8x-seg.pt")),
    "yolov8x-oiv7": YOLO(models_directory.joinpath("yolov8x-oiv7.pt"))
}

# Try loading an image with a model
try:
    image = Image.open(images_directory.joinpath("0.png"))
    image = np.asarray(image)
    yolo_models["yolov8x"].predict(image)
except:
    pass


def go_to_recaptcha_iframe(driver, iframe_xpath):
    driver.switch_to.default_content()
    recaptcha_iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, iframe_xpath))
    )
    driver.switch_to.frame(recaptcha_iframe)


def get_target_num(target_text):
    target_mapping = {
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
        "taxi": 1003
    }

    for key, value in target_mapping.items():
        if re.search(key, target_text) is not None:
            return value

    return 1000


def get_answers(target_num, timestamp):
    image = Image.open(images_directory.joinpath(f"0-{timestamp}.png"))
    image = np.asarray(image)

    if target_num == 1001:
        result = yolo_models["crosswalk"].predict(image)
        target_num = 0
    elif target_num == 1002:
        result = yolo_models["yolov8x-oiv7"].predict(image)
        target_num = 489
    elif target_num == 1003:
        result = yolo_models["yolov8x-oiv7"].predict(image)
        target_num = 522
    else:
        result = yolo_models["yolov8x"].predict(image)

    target_index = [i for i, num in enumerate(
        result[0].boxes.cls) if num == target_num]

    answers = set()

    boxes = result[0].boxes.data
    for i in target_index:
        target_box = boxes[i]
        xc, yc = (target_box[0] + target_box[2]) / \
            2, (target_box[1] + target_box[3]) / 2

        x_pos = int(xc / (image.shape[1] / 3))
        y_pos = int(yc / (image.shape[0] / 3))

        point_num = y_pos * 3 + x_pos + 1

        answers.add(point_num)

    return list(answers)


def get_all_captcha_img_urls(driver):
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))

    img_urls = []
    for img in images:
        img_urls.append(img.get_attribute("src"))

    return img_urls


def download_img(name, url, timestamp):
    response = requests.get(url, stream=True)
    with open(images_directory.joinpath(f'{name}-{timestamp}.png'), 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response


def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    images = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located(
        (By.XPATH, '//div[@id="rc-imageselect-target"]//img')))
    img_urls = []

    for img in images:
        try:
            img_urls.append(img.get_attribute("src"))
        except:
            is_new = False
            return is_new, img_urls

    index_common = []
    for answer in answers:
        if img_urls[answer - 1] == before_img_urls[answer - 1]:
            index_common.append(answer)

    if len(index_common) >= 1:
        is_new = False
        return is_new, img_urls
    else:
        is_new = True
        return is_new, img_urls


def paste_new_img_on_main_img(main, new, loc, timestamp):
    paste = np.copy(main)

    section_sizes = {
        1: (0, 0), 2: (0, 1), 3: (0, 2),
        4: (1, 0), 5: (1, 1), 6: (1, 2),
        7: (2, 0), 8: (2, 1), 9: (2, 2)
    }

    section_row, section_col = section_sizes.get(loc, (0, 0))
    height, width = paste.shape[0] // 3, paste.shape[1] // 3
    start_row, start_col = section_row * height, section_col * width
    paste[start_row:start_row + height, start_col:start_col + width] = new

    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(images_directory.joinpath(f'0-{timestamp}.png')), paste)


def get_occupied_cells(vertices):
    occupied_cells = set()
    rows, cols = zip(*[((v - 1) // 4, (v - 1) % 4) for v in vertices])

    for i in range(min(rows), max(rows) + 1):
        for j in range(min(cols), max(cols) + 1):
            occupied_cells.add(4 * i + j + 1)

    return sorted(list(occupied_cells))


def detect_car_locations(masked_image, num_rows=4, num_columns=4, threshold=100):
    height, width, _ = masked_image.shape
    row_step = height // num_rows
    col_step = width // num_columns
    car_locations = []

    for i in range(num_rows):
        for j in range(num_columns):
            row_start = i * row_step
            row_end = (i + 1) * row_step
            col_start = j * col_step
            col_end = (j + 1) * col_step
            region = masked_image[row_start:row_end, col_start:col_end]
            white_pixel_count = np.sum(region > 0)
            if white_pixel_count > threshold:
                car_locations.append((i, j))

    return car_locations


def convert_to_position_indices(car_locations, num_rows=4, num_columns=4):
    position_indices = []
    for i, j in car_locations:
        position = i * num_columns + j + 1
        position_indices.append(position)
    return position_indices


def get_answers_4(target_num, timestamp):
    image = Image.open(images_directory.joinpath(f"0-{timestamp}.png"))
    image = np.asarray(image)

    if target_num < 1000:
        result_seg = yolo_models["yolov8x-seg"].predict(image)

        target_index = []
        count = 0
        for num in result_seg[0].boxes.cls:
            if num == target_num:
                target_index.append(count)
            count += 1

        answers = []

        for i in target_index:
            if (result_seg[0].masks is not None):
                mask_raw = result_seg[0].masks[i].cpu(
                ).data.numpy().transpose(1, 2, 0)
                mask_3channel = cv2.merge((mask_raw, mask_raw, mask_raw))
                h2, w2, c2 = result_seg[0].orig_img.shape
                mask = cv2.resize(mask_3channel, (w2, h2))
                hsv = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
                lower_black = np.array([0, 0, 0])
                upper_black = np.array([0, 0, 1])
                mask = cv2.inRange(mask, lower_black, upper_black)
                mask = cv2.bitwise_not(mask)
                masked = cv2.bitwise_and(
                    result_seg[0].orig_img, result_seg[0].orig_img, mask=mask)
                car_locations = detect_car_locations(
                    masked, num_rows=4, num_columns=4, threshold=100)
                position_indices = convert_to_position_indices(car_locations)
                for indice in position_indices:
                    answers.append(indice)

        answers = sorted(list(answers))
        return list(set(answers))
    else:
        if target_num == 1001:
            result = yolo_models["crosswalk"].predict(image)
            target_num = 0
        elif target_num == 1002:
            result = yolo_models["yolov8x-oiv7"].predict(image)
            target_num = 489
        elif target_num == 1003:
            result = yolo_models["yolov8x-oiv7"].predict(image)
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
            p1, p2 = (int(target_box[0]), int(target_box[1])), (int(
                target_box[2]), int(target_box[3]))
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
                p1, p2 = (int(target_box[0]), int(target_box[1])), (int(
                    target_box[2]), int(target_box[3]))
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


def solve_recaptcha(driver):
    go_to_recaptcha_iframe(driver, '//iframe[@title="reCAPTCHA"]')

    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
        (By.XPATH, '//div[@class="recaptcha-checkbox-border"]'))).click()

    go_to_recaptcha_iframe(driver, '//iframe[contains(@title, "challenge")]')

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    random_num = random.randint(100000, 999999)
    timestamp = f"{timestamp}_{random_num}"

    while True:
        try:
            while True:
                solved = False
                for i in range(200):
                    try:
                        go_to_recaptcha_iframe(
                            driver, '//iframe[contains(@title, "challenge")]')
                        reload = WebDriverWait(driver, 0.1).until(
                            EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                        solved = False
                        break
                    except:
                        go_to_recaptcha_iframe(
                            driver, '//iframe[@title="reCAPTCHA"]')
                        if WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//span[@id="recaptcha-anchor"]'))).get_attribute("aria-checked") == 'true':
                            solved = True
                            break
                        else:
                            solved = False
                if solved:
                    break

                reload = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.ID, 'recaptcha-reload-button')))
                title_wrapper = WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.ID, 'rc-imageselect')))

                target_text = title_wrapper.find_element(
                    By.XPATH, './/strong').text
                target_num = get_target_num(target_text)

                if target_num == 1000:
                    print("Skipping")
                    reload.click()
                elif "squares" in title_wrapper.text:
                    print("Square captcha found....")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0], timestamp)
                    answers = get_answers_4(target_num, timestamp)
                    if len(answers) >= 1 and len(answers) < 16:
                        captcha = "squares"
                        break
                    else:
                        reload.click()
                elif "none" in title_wrapper.text:
                    print("Found a 3x3 dynamic captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    if len(set(img_urls)) == 1:
                        download_img(0, img_urls[0], timestamp)
                        answers = get_answers(target_num, timestamp)
                        if len(answers) > 2:
                            captcha = "dynamic"
                            break
                        else:
                            reload.click()
                    else:
                        reload.click()
                else:
                    print("Found a 3x3 one-time selection captcha")
                    img_urls = get_all_captcha_img_urls(driver)
                    download_img(0, img_urls[0], timestamp)
                    answers = get_answers(target_num, timestamp)
                    if len(answers) > 2:
                        captcha = "selection"
                        break
                    else:
                        reload.click()
                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                    (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')))

            if solved:
                print("Solved")
                driver.switch_to.default_content()
                break
            if captcha == "dynamic":
                for answer in answers:
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                while True:
                    before_img_urls = img_urls
                    while True:
                        is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                            answers, before_img_urls, driver)
                        if is_new:
                            break

                    new_img_index_urls = []
                    for answer in answers:
                        new_img_index_urls.append(answer - 1)

                    for index in new_img_index_urls:
                        download_img(index + 1, img_urls[index], timestamp)
                    while True:
                        try:
                            for answer in answers:
                                main_img = Image.open(
                                    images_directory.joinpath(f"0-{timestamp}.png"))
                                new_img = Image.open(images_directory.joinpath(
                                    f"{answer}-{timestamp}.png"))
                                location = answer
                                paste_new_img_on_main_img(
                                    main_img, new_img, location, timestamp)
                            break
                        except:
                            while True:
                                is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                    answers, before_img_urls, driver)
                                if is_new:
                                    break
                            new_img_index_urls = []
                            for answer in answers:
                                new_img_index_urls.append(answer - 1)

                            for index in new_img_index_urls:
                                download_img(
                                    index + 1, img_urls[index], timestamp)

                    answers = get_answers(target_num, timestamp)

                    if len(answers) >= 1:
                        for answer in answers:
                            WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                                (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()
                    else:
                        break
            elif captcha == "selection" or captcha == "squares":
                for answer in answers:
                    WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                        (By.XPATH, f'(//div[@id="rc-imageselect-target"]//td)[{answer}]'))).click()

            verify = WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                (By.ID, "recaptcha-verify-button")))
            verify.click()

            for i in range(200):
                try:
                    go_to_recaptcha_iframe(
                        driver, '//iframe[contains(@title, "challenge")]')
                    WebDriverWait(driver, 0.1).until(
                        EC.presence_of_element_located(
                            (By.XPATH, '//button[@id="recaptcha-verify-button" and not(contains(@class, "rc-button-default-disabled"))]'))
                    )
                    solved = False
                    break
                except:
                    go_to_recaptcha_iframe(
                        driver, '//iframe[@title="reCAPTCHA"]')
                    if WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.XPATH, '//span[@id="recaptcha-anchor"]'))).get_attribute("aria-checked") == 'true':
                        solved = True
                        break
                    else:
                        solved = False
            if solved:
                print("Solved")
                list_images_directory = listdir(images_directory)
                for image in list_images_directory:
                    if re.search(timestamp, image) is not None:
                        remove(images_directory.joinpath(image))

                driver.switch_to.default_content()
                break
            else:
                go_to_recaptcha_iframe(
                    driver, '//iframe[contains(@title, "challenge")]')

        except Exception as e:
            print(e)
            try:
                go_to_recaptcha_iframe(driver, '//iframe[@title="reCAPTCHA"]')

                WebDriverWait(driver, 10).until(EC.element_to_be_clickable(
                    (By.XPATH, '//div[@class="recaptcha-checkbox-border"]'))).click()

                go_to_recaptcha_iframe(driver, '//iframe[contains(@title, "challenge")]')
            except:
                ...