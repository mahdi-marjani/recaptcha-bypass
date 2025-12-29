from .solver import RecaptchaSolver
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager


if __name__ == "__main__":
    driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
    driver.get("https://google.com/recaptcha/api2/demo")

    solver = RecaptchaSolver(driver)
    solver.solve()
    
    print("reCAPTCHA solved!")
    driver.quit()
