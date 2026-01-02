import os
import re
import random
from datetime import datetime
from PIL import Image

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .config import get_target_num, IMAGES_DIRECTORY
from .utils import (
    switch_to_recaptcha_frame,
    get_all_image_urls,
    download_image,
    get_new_dynamic_image_urls,
    paste_image_on_main,
)
from .detection import detect_cells_3x3, detect_cells_4x4


class RecaptchaSolver:
    def __init__(self, driver):
        self.driver = driver
        self.timestamp = (
            datetime.now().strftime("%Y%m%d%H%M%S")
            + f"_{random.randint(100000, 999999)}"
        )

    # =========================
    # Public API
    # =========================

    def solve(self):
        self._click_checkbox()
        self._solve_challenge()

    # =========================
    # High-level flow
    # =========================

    def _solve_challenge(self):
        while True:
            try:
                solved = self._wait_until_challenge_ready_or_solved()
                if solved:
                    self._finalize()
                    break

                captcha_type, target_num, answers = self._analyze_challenge()

                if captcha_type is None:
                    self._reload()
                    continue

                if captcha_type == "dynamic":
                    self._solve_dynamic(target_num, answers)
                else:
                    self._click_answers(answers)

                if self._verify():
                    self._finalize()
                    break

            except Exception as e:
                print(e)
                self._recover_from_error()

    # =========================
    # Initial checkbox
    # =========================

    def _click_checkbox(self):
        switch_to_recaptcha_frame(self.driver, '//iframe[@title="reCAPTCHA"]')
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable(
                (By.XPATH, '//div[@class="recaptcha-checkbox-border"]')
            )
        ).click()
        self._switch_to_challenge_frame()

    # =========================
    # Challenge state handling
    # =========================

    def _wait_until_challenge_ready_or_solved(self):
        for _ in range(200):
            try:
                self._switch_to_challenge_frame()
                WebDriverWait(self.driver, 0.1).until(
                    EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
                )
                return False
            except Exception:
                if self._is_checkbox_checked():
                    return True
        return False

    def _analyze_challenge(self):
        self._switch_to_challenge_frame()

        reload_btn = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
        )
        title_wrapper = WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.ID, "rc-imageselect"))
        )

        target_text = title_wrapper.find_element(By.XPATH, ".//strong").text
        target_num = get_target_num(target_text)

        if target_num == 1000:
            return None, None, None

        if "squares" in title_wrapper.text:
            return self._handle_4x4(target_num)

        if "none" in title_wrapper.text:
            return self._handle_dynamic_3x3(target_num)

        return self._handle_static_3x3(target_num)

    # =========================
    # CAPTCHA type handlers
    # =========================

    def _handle_4x4(self, target_num):
        img_urls = get_all_image_urls(self.driver)
        download_image(0, img_urls[0], self.timestamp)

        answers = detect_cells_4x4(target_num, self.timestamp)
        if 1 <= len(answers) < 16:
            return "squares", target_num, answers

        return None, None, None

    def _handle_dynamic_3x3(self, target_num):
        img_urls = get_all_image_urls(self.driver)
        if len(set(img_urls)) != 1:
            return None, None, None

        download_image(0, img_urls[0], self.timestamp)
        answers = detect_cells_3x3(target_num, self.timestamp)

        if len(answers) > 2:
            return "dynamic", target_num, answers

        return None, None, None

    def _handle_static_3x3(self, target_num):
        img_urls = get_all_image_urls(self.driver)
        download_image(0, img_urls[0], self.timestamp)

        answers = detect_cells_3x3(target_num, self.timestamp)
        if len(answers) > 2:
            return "selection", target_num, answers

        return None, None, None

    # =========================
    # Solvers
    # =========================

    def _solve_dynamic(self, target_num, answers):
        self._click_answers(answers)

        while True:
            old_urls = get_all_image_urls(self.driver)
            is_new, new_urls = self._wait_for_new_dynamic_images(
                answers, old_urls
            )

            self._download_dynamic_images(answers, new_urls)
            self._merge_dynamic_images(answers)

            answers = detect_cells_3x3(target_num, self.timestamp)
            if not answers:
                break

            self._click_answers(answers)

    # =========================
    # Actions
    # =========================

    def _click_answers(self, answers):
        for answer in answers:
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable(
                    (
                        By.XPATH,
                        f'(//div[@id="rc-imageselect-target"]//td)[{answer}]',
                    )
                )
            ).click()

    def _verify(self):
        verify = WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "recaptcha-verify-button"))
        )
        verify.click()

        for _ in range(200):
            try:
                self._switch_to_challenge_frame()
                WebDriverWait(self.driver, 0.1).until(
                    EC.presence_of_element_located(
                        (
                            By.XPATH,
                            '//button[@id="recaptcha-verify-button" and not(contains(@class, "rc-button-default-disabled"))]',
                        )
                    )
                )
                return False
            except Exception:
                if self._is_checkbox_checked():
                    self._cleanup_images()
                    return True

        return False

    # =========================
    # Dynamic helpers
    # =========================

    def _wait_for_new_dynamic_images(self, answers, old_urls):
        while True:
            is_new, new_urls = get_new_dynamic_image_urls(
                answers, old_urls, self.driver
            )
            if is_new:
                return is_new, new_urls

    def _download_dynamic_images(self, answers, img_urls):
        for answer in answers:
            idx = answer - 1
            download_image(answer, img_urls[idx], self.timestamp)

    def _merge_dynamic_images(self, answers):
        while True:
            try:
                for answer in answers:
                    main_img = Image.open(
                        IMAGES_DIRECTORY / f"0-{self.timestamp}.png"
                    )
                    new_img = Image.open(
                        IMAGES_DIRECTORY / f"{answer}-{self.timestamp}.png"
                    )
                    paste_image_on_main(
                        main_img, new_img, answer, self.timestamp
                    )
                return
            except Exception:
                continue

    # =========================
    # Frame / state utils
    # =========================

    def _switch_to_challenge_frame(self):
        switch_to_recaptcha_frame(
            self.driver,
            '//iframe[contains(@title, "challenge") and contains(@title, "recaptcha")]',
        )

    def _is_checkbox_checked(self):
        switch_to_recaptcha_frame(self.driver, '//iframe[@title="reCAPTCHA"]')
        return (
            WebDriverWait(self.driver, 10)
            .until(
                EC.presence_of_element_located(
                    (By.XPATH, '//span[@id="recaptcha-anchor"]')
                )
            )
            .get_attribute("aria-checked")
            == "true"
        )

    def _reload(self):
        self._switch_to_challenge_frame()
        WebDriverWait(self.driver, 10).until(
            EC.element_to_be_clickable((By.ID, "recaptcha-reload-button"))
        ).click()

    def _recover_from_error(self):
        try:
            self._click_checkbox()
        except Exception:
            pass

    def _finalize(self):
        self.driver.switch_to.default_content()

    # =========================
    # Cleanup
    # =========================

    def _cleanup_images(self):
        for image in os.listdir(IMAGES_DIRECTORY):
            if re.search(self.timestamp, image):
                os.remove(IMAGES_DIRECTORY / image)
