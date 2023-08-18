import pyautogui
import json
import time
from selenium.webdriver.chrome.options import Options
import glob
from bs4 import BeautifulSoup as bs
import os
import requests
import selenium 
from  newsfetch.news import newspaper

#%% get html
def get_questions_html(driver, url, sleep_time=0):
    print( 'Loading questions page...')
    driver.get(url)
    time.sleep(sleep_time)
    return driver

#%%
with open("./Europe_Usa.json_cleaned_final.json") as f:
    data = json.load(f)
#%%

driver = selenium.webdriver.Chrome('/home/stephan/chromedriver')
driver = get_questions_html(driver, 'https://www.google.com')
pyautogui.moveTo(650,850)
pyautogui.click()
time.sleep(1)

for item in data.values():
    search = '+'.join(item['Name'].split())
    search = 'https://www.google.com/search?q=' + search
    driver = get_questions_html(driver, search)
    pyautogui.moveTo(275,265)
    pyautogui.click()
    time.sleep(2)
    text = driver.page_source
    content = bs(text).find_all('a', href=True)
    content = [a['href'] for a in content if not 'google' in a['href'] and not 'search?q=' in a['href']]
    news = [newspaper(c).get_dict for c in content]
    print(news)
    break
#%%
print(news[5])
