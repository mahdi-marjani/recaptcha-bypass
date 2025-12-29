import random
from datetime import datetime
from os import listdir, remove
import re
from PIL import Image
from .config import get_target_num, IMAGES_DIRECTORY
from .utils import (go_to_recaptcha_iframe, get_all_captcha_img_urls, download_img,
                    get_all_new_dynamic_captcha_img_urls, paste_new_img_on_main_img)
from .detection import get_answers, get_answers_4
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class RecaptchaSolver:
    def __init__(self, driver):
        self.driver = driver
        self.timestamp = datetime.now().strftime("%Y%m%d%H%M%S") + f"_{random.randint(100000, 999999)}"

    def solve(self):
        go_to_recaptcha_iframe(self.driver, '//iframe[@title="reCAPTCHA"]')

        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
            )
        ).click()

        go_to_recaptcha_iframe(
            self.driver,
            '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
        )

        while True:
            try:
                while True:
                    solved = False
                    for i in range(200):
                        try:
                            go_to_recaptcha_iframe(
                                self.driver,
                                '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
                            )
                            reload = WebDriverWait(self.driver, 0.1).until(
                                EC.element_to_be_clickable(
                                    (By.ID, "recaptcha-reload-button")
                                )
                            )
                            solved = False
                            break
                        except:
                            go_to_recaptcha_iframe(self.driver, '//iframe[@title="reCAPTCHA"]')
                            if (
                                WebDriverWait(self.driver, 10)
                                .until(
                                    EC.presence_of_element_located(
                                        (By.XPATH, '//span[@id="recaptcha-anchor"]')
                                    )
                                )
                                .get_attribute("aria-checked")
                                == "true"
                            ):
                                solved = True
                                break
                            else:
                                solved = False
                    if solved:
                        break

                    reload = WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
                    )
                    title_wrapper = WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.ID, "rc-imageselect"))
                    )

                    target_text = title_wrapper.find_element(By.XPATH, ".//strong").text
                    target_num = get_target_num(target_text)

                    if target_num == 1000:
                        print("Skipping")
                        reload.click()
                    elif "squares" in title_wrapper.text:
                        print("Square captcha found....")
                        img_urls = get_all_captcha_img_urls(self.driver)
                        download_img(0, img_urls[0], self.timestamp)
                        answers = get_answers_4(target_num, self.timestamp)
                        if len(answers) >= 1 and len(answers) < 16:
                            captcha = "squares"
                            break
                        else:
                            reload.click()
                    elif "none" in title_wrapper.text:
                        print("Found a 3x3 dynamic captcha")
                        img_urls = get_all_captcha_img_urls(self.driver)
                        if len(set(img_urls)) == 1:
                            download_img(0, img_urls[0], self.timestamp)
                            answers = get_answers(target_num, self.timestamp)
                            if len(answers) > 2:
                                captcha = "dynamic"
                                break
                            else:
                                reload.click()
                        else:
                            reload.click()
                    else:
                        print("Found a 3x3 one-time selection captcha")
                        img_urls = get_all_captcha_img_urls(self.driver)
                        download_img(0, img_urls[0], self.timestamp)
                        answers = get_answers(target_num, self.timestamp)
                        if len(answers) > 2:
                            captcha = "selection"
                            break
                        else:
                            reload.click()
                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, '(//div[@id="rc-imageselect-target"]//td)[1]')
                        )
                    )

                if solved:
                    print("Solved")
                    self.driver.switch_to.default_content()
                    break
                if captcha == "dynamic":
                    for answer in answers:
                        WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable(
                                (
                                    By.XPATH,
                                    f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                                )
                            )
                        ).click()
                    while True:
                        before_img_urls = img_urls
                        while True:
                            is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                answers, before_img_urls, self.driver
                            )
                            if is_new:
                                break

                        new_img_index_urls = []
                        for answer in answers:
                            new_img_index_urls.append(answer - 1)

                        for index in new_img_index_urls:
                            download_img(index + 1, img_urls[index], self.timestamp)
                        while True:
                            try:
                                for answer in answers:
                                    main_img = Image.open(
                                        IMAGES_DIRECTORY.joinpath(f"0-{self.timestamp}.png")
                                    )
                                    new_img = Image.open(
                                        IMAGES_DIRECTORY.joinpath(
                                            f"{answer}-{self.timestamp}.png"
                                        )
                                    )
                                    location = answer
                                    paste_new_img_on_main_img(
                                        main_img, new_img, location, self.timestamp
                                    )
                                break
                            except:
                                while True:
                                    is_new, img_urls = get_all_new_dynamic_captcha_img_urls(
                                        answers, before_img_urls, self.driver
                                    )
                                    if is_new:
                                        break
                                new_img_index_urls = []
                                for answer in answers:
                                    new_img_index_urls.append(answer - 1)

                                for index in new_img_index_urls:
                                    download_img(index + 1, img_urls[index], self.timestamp)

                        answers = get_answers(target_num, self.timestamp)

                        if len(answers) >= 1:
                            for answer in answers:
                                WebDriverWait(self.driver, 10).until(
                                    EC.element_to_be_clickable(
                                        (
                                            By.XPATH,
                                            f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                                        )
                                    )
                                ).click()
                        else:
                            break
                elif captcha == "selection" or captcha == "squares":
                    for answer in answers:
                        WebDriverWait(self.driver, 10).until(
                            EC.element_to_be_clickable(
                                (
                                    By.XPATH,
                                    f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                                )
                            )
                        ).click()

                verify = WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.ID, "recaptcha-verify-button"))
                )
                verify.click()

                for i in range(200):
                    try:
                        go_to_recaptcha_iframe(
                            self.driver,
                            '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
                        )
                        WebDriverWait(self.driver, 0.1).until(
                            EC.presence_of_element_located(
                                (
                                    By.XPATH,
                                    '//button[@id="recaptcha-verify-button" and not(contains(@class, "rc-button-default-disabled"))]',
                                )
                            )
                        )
                        solved = False
                        break
                    except:
                        go_to_recaptcha_iframe(self.driver, '//iframe[@title="reCAPTCHA"]')
                        if (
                            WebDriverWait(self.driver, 10)
                            .until(
                                EC.presence_of_element_located(
                                    (By.XPATH, '//span[@id="recaptcha-anchor"]')
                                )
                            )
                            .get_attribute("aria-checked")
                            == "true"
                        ):
                            solved = True
                            break
                        else:
                            solved = False
                if solved:
                    print("Solved")
                    self._cleanup_images()

                    self.driver.switch_to.default_content()
                    break
                else:
                    go_to_recaptcha_iframe(
                        self.driver,
                        '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
                    )

            except Exception as e:
                print(e)
                try:
                    go_to_recaptcha_iframe(self.driver, '//iframe[@title="reCAPTCHA"]')

                    WebDriverWait(self.driver, 10).until(
                        EC.element_to_be_clickable(
                            (By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
                        )
                    ).click()

                    go_to_recaptcha_iframe(
                        self.driver,
                        '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
                    )
                except:
                    ...


    def _cleanup_images(self):
        # کد remove تصاویر با timestamp
        list_images = listdir(IMAGES_DIRECTORY)
        for image in list_images:
            if re.search(self.timestamp, image):
                remove(IMAGES_DIRECTORY / image)