from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from .utils import (
    switch_to_recaptcha_frame,
    get_all_image_urls,
    get_image_array,
    get_new_dynamic_image_urls,
    paste_image_on_main,
)

from .local_ai import is_model_available, detect


class RecaptchaSolver:
    def __init__(self, driver):
        self.driver = driver

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

                captcha_type, target_text, answers, main_image_array = self._analyze_challenge()

                if captcha_type is None:
                    self._reload()
                    continue

                if captcha_type == "dynamic":
                    self._solve_dynamic(target_text, answers, main_image_array)
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

        if not is_model_available(target_text):
            return None, None, None, None

        if "squares" in title_wrapper.text:
            return self._handle_4x4(target_text)

        if "none" in title_wrapper.text:
            return self._handle_dynamic_3x3(target_text)

        return self._handle_static_3x3(target_text)

    # =========================
    # CAPTCHA type handlers
    # =========================

    def _handle_4x4(self, target_text):
        img_urls = get_all_image_urls(self.driver)
        main_image_array = get_image_array(img_urls[0])

        answers = detect(main_image_array, "4x4", target_text)
        if 1 <= len(answers) < 16:
            return "squares", target_text, answers, main_image_array

        return None, None, None, None

    def _handle_dynamic_3x3(self, target_text):
        img_urls = get_all_image_urls(self.driver)
        if len(set(img_urls)) != 1:
            return None, None, None

        main_image_array = get_image_array(img_urls[0])
        answers = detect(main_image_array, "3x3", target_text)

        if len(answers) > 2:
            return "dynamic", target_text, answers, main_image_array

        return None, None, None, None

    def _handle_static_3x3(self, target_text):
        img_urls = get_all_image_urls(self.driver)
        main_image_array = get_image_array(img_urls[0])

        answers = detect(main_image_array, "3x3", target_text)
        if len(answers) > 2:
            return "selection", target_text, answers, main_image_array

        return None, None, None, None

    # =========================
    # Solvers
    # =========================

    def _solve_dynamic(self, target_text, answers, main_image_array):
        self._click_answers(answers)

        while True:
            old_urls = get_all_image_urls(self.driver)
            is_new, new_urls = self._wait_for_new_dynamic_images(
                answers, old_urls
            )

            new_images_array = self._download_dynamic_images(answers, new_urls)
            main_image_array = self._merge_dynamic_images(answers, main_image_array, new_images_array)

            answers = detect(main_image_array, "3x3", target_text)
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
        new_images_array = {}
        for answer in answers:
            idx = answer - 1
            new_images_array[answer] = get_image_array(img_urls[idx])
        
        return new_images_array

    def _merge_dynamic_images(self, answers, main_image_array, new_images_array):
        while True:
            try:
                for answer in answers:
                    new_image_array = new_images_array[answer]
                    main_image_array = paste_image_on_main(main_image_array, new_image_array, answer)
                return main_image_array
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