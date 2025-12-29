import shutil
import requests
import cv2
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .config import IMAGES_DIRECTORY

def go_to_recaptcha_iframe(driver, iframe_xpath):
    driver.switch_to.default_content()
    recaptcha_iframe = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, iframe_xpath))
    )
    driver.switch_to.frame(recaptcha_iframe)

def download_img(name, url, timestamp):
    response = requests.get(url, stream=True)
    with open(IMAGES_DIRECTORY / f"{name}-{timestamp}.png", "wb") as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response

def paste_new_img_on_main_img(main, new, loc, timestamp):
    paste = np.copy(main)

    section_sizes = {
        1: (0, 0),
        2: (0, 1),
        3: (0, 2),
        4: (1, 0),
        5: (1, 1),
        6: (1, 2),
        7: (2, 0),
        8: (2, 1),
        9: (2, 2),
    }

    section_row, section_col = section_sizes.get(loc, (0, 0))
    height, width = paste.shape[0] // 3, paste.shape[1] // 3
    start_row, start_col = section_row * height, section_col * width
    paste[start_row : start_row + height, start_col : start_col + width] = new

    paste = cv2.cvtColor(paste, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(IMAGES_DIRECTORY.joinpath(f"0-{timestamp}.png")), paste)

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

def get_all_captcha_img_urls(driver):
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )

    img_urls = []
    for img in images:
        img_urls.append(img.get_attribute("src"))

    return img_urls

def get_all_new_dynamic_captcha_img_urls(answers, before_img_urls, driver):
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )
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