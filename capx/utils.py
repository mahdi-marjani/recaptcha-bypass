import io
import requests
import numpy as np
from PIL import Image
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


def switch_to_recaptcha_frame(driver, frame_xpath):
    """Switch to the reCAPTCHA frame."""
    driver.switch_to.default_content()
    frame = WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, frame_xpath))
    )
    driver.switch_to.frame(frame)


def get_image_array(url):
    """Download an image from URL and return it as a numpy array."""
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(io.BytesIO(response.content))
        return np.array(image)
    else:
        raise ValueError(f"Failed to download image from {url}")


def paste_image_on_main(main_img, new_img, position):
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
    
    return main


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