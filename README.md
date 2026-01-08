# reCAPTCHA v2 Image Solver

reCAPTCHA v2 Image Solver using YOLO and Selenium.

https://github.com/user-attachments/assets/22308be7-3a90-4757-8799-b47008b32bf0

## How it works
- Spots objects (cars, buses, crosswalks, etc.) in the images
- Clicks the right tiles for you
- Works with 3x3, 4x4, static, and dynamic challenges

### 1. virtual environment (optional but recommended)
```bash
python -m venv venv
# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate
```

### 2. Install
```bash
pip install capx
```

### 3. Quick example
```python
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from webdriver_manager.firefox import GeckoDriverManager
from capx.solver import RecaptchaSolver

driver = webdriver.Firefox(service=Service(GeckoDriverManager().install()))
driver.get("https://www.google.com/recaptcha/api2/demo")

solver = RecaptchaSolver(driver)
solver.solve()  # Done!

print("reCAPTCHA solved!")
input("Press Enter to quit...")
driver.quit()
```

Have fun! 🚀
