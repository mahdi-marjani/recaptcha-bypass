from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from recaptcha_bypass import solve_recaptcha

driver = webdriver.Firefox(service=Service(
    GeckoDriverManager().install()))

driver.get('https://google.com/recaptcha/api2/demo')

solve_recaptcha(driver)

input('press Enter...')
