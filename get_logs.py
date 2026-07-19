import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

options = Options()
options.add_argument("--headless")
options.add_argument("--disable-gpu")
options.set_capability('goog:loggingPrefs', {'browser': 'ALL'})

driver = webdriver.Chrome(options=options)
driver.get("http://127.0.0.1:8001")
time.sleep(2)

logs = driver.get_log('browser')
for log in logs:
    print(f"[{log['level']}] {log['message']}")

driver.quit()
