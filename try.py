from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By


chromeOptions = Options()
chromeOptions.add_argument("--ignore-certificate-errors")
chromeOptions.add_argument("--incognito")
# chromeOptions.add_argument("--headless")

driver = webdriver.Chrome("./chromedriver_linux64/chromedriver_73", options=chromeOptions)

driver.get(str('https://www.youtube.com/watch?v=9JmpegI1Wvc&t=2268s'))