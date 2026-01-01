import shutil
import requests
import cv2
import numpy as np
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from .config import IMAGES_DIRECTORY

def switch_to_recaptcha_frame(driver, frame_xpath):
    """Switch to the reCAPTCHA frame."""
    driver.switch_to.default_content()
    frame = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, frame_xpath))
    )
    driver.switch_to.frame(frame)


def download_image(name, url, timestamp):
    """Download an image from URL and save it."""
    response = requests.get(url, stream=True)
    with open(IMAGES_DIRECTORY / f"{name}-{timestamp}.png", "wb") as file:
        shutil.copyfileobj(response.raw, file)
    del response


def paste_image_on_main(main_img, new_img, position, timestamp):
    """Paste a new image onto the main one at a position.

    Positions map to a 3x3 grid like this:
    +---+---+---+
    | 1 | 2 | 3 |
    +---+---+---+
    | 4 | 5 | 6 |
    +---+---+---+
    | 7 | 8 | 9 |
    +---+---+---+
    """
    main = np.copy(main_img)

    section_map = {
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
    
    row, col = section_map[position]
    height, width = main.shape[0] // 3, main.shape[1] // 3
    
    start_row = row * height
    start_col = col * width
    
    main[start_row : start_row + height, start_col : start_col + width] = new_img
    
    main = cv2.cvtColor(main, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(IMAGES_DIRECTORY / f"0-{timestamp}.png"), main)


def find_filled_cells(corners):
    """Find all cells filled by the given corner numbers in a 4x4 grid.

    It takes corner numbers (like [1,4,13,16]) and returns all cells inside the rectangle they form.
    Simple: find min/max rows/cols, then fill the square.
    """

    # Get row and col for each corner (0-based)
    positions = [((v - 1) // 4, (v - 1) % 4) for v in corners]
    rows = [pos[0] for pos in positions]
    cols = [pos[1] for pos in positions]

    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    filled = []
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell_num = (r * 4) + c + 1
            filled.append(cell_num)

    return sorted(set(filled))  # Remove duplicates if any


def find_object_locations(image, rows=4, cols=4, min_pixels=100):
    """Find grid spots with objects (like cars or anything) by checking pixel count.

    Splits image into rows x cols grid, checks if each spot has > min_pixels bright pixels.
    Returns list of (row, col) spots with objects. Rows/cols start from 0.
    """
    height, width, _ = image.shape
    row_size = height // rows
    col_size = width // cols

    locations = []
    for r in range(rows):
        for c in range(cols):
            spot = image[
                r * row_size : (r + 1) * row_size,
                c * col_size : (c + 1) * col_size
            ]
            bright_count = np.sum(spot > 0)
            if bright_count > min_pixels:
                locations.append((r, c))

    return locations


def locations_to_numbers(locations, cols=4):
    """Turn (row, col) locations into simple numbers (1-based) for a grid.

    Example grid numbering (for 4 cols):
    +----+----+----+----+
    |  1 |  2 |  3 |  4 |
    +----+----+----+----+
    |  5 |  6 |  7 |  8 |
    +----+----+----+----+
    |  9 | 10 | 11 | 12 |
    +----+----+----+----+
    | 13 | 14 | 15 | 16 |
    +----+----+----+----+

    So (0,0) -> 1, (1,2) -> 7, etc.
    """
    numbers = []
    for row, col in locations:
        num = (row * cols) + col + 1
        numbers.append(num)
    return numbers


def get_all_image_urls(driver):
    """Get all image URLs from the CAPTCHA grid."""
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )
    return [img.get_attribute("src") for img in images]


def get_new_dynamic_image_urls(answers, old_urls, driver):
    """Check for new dynamic CAPTCHA images and return if changed."""
    images = WebDriverWait(driver, 10).until(
        EC.presence_of_all_elements_located(
            (By.XPATH, '//div[@id="rc-imageselect-target"]//img')
        )
    )
    new_urls = []

    for img in images:
        try:
            new_urls.append(img.get_attribute("src"))
        except:
            is_new = False
            return is_new, new_urls

    same_count = 0
    for answer in answers:
        if new_urls[answer - 1] == old_urls[answer - 1]:
            same_count += 1

    if same_count > 0:
        is_new = False
        return is_new, new_urls
    else:
        is_new = True
        return is_new, new_urls
