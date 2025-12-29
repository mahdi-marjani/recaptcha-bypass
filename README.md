# reCAPTCHA v2 Image Solver

Simple Python tool to automatically solve Google's reCAPTCHA v2 "select all squares" challenges using Selenium + YOLO.

**For educational and testing purposes only.**

https://github.com/hooshmang/recaptcha-bypass/assets/129745867/024b82eb-1dc7-4f3d-a5d4-598dede2f5e4

## How it works
- Detects objects (cars, buses, crosswalks, etc.) in captcha images
- Clicks the correct tiles automatically
- Handles 3x3, 4x4, static and dynamic challenges

## Installation

```bash
git clone https://github.com/mahdi-marjani/recaptcha-bypass.git
cd recaptcha-bypass
pip install -r requirements.txt
```

## Usage

```python
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from src.solver import RecaptchaSolver

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
driver.get("https://www.google.com/recaptcha/api2/demo")

solver = RecaptchaSolver(driver)
solver.solve()  # That's it

input("Press Enter to close...")
driver.quit()
```

Works with Firefox too — see `src/main.py` for examples.

Enjoy! 🚀
