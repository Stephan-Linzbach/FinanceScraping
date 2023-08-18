import time
import json
from bs4 import BeautifulSoup as sp
import selenium.webdriver
from selenium.webdriver.chrome.options import Options


# Use Selenium driver to get questions html page from url.
def get_questions_html(url, sleep_time):
    print('Loading questions page...')
    chromeOptions = Options()
    chromeOptions.headless = True
    driver = selenium.webdriver.Chrome('/home/stephan/chromedriver', options=chromeOptions)
    driver.get(url)
    """
    for i in range(1, times_to_scroll):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(sleep_time)
    """
    time.sleep(sleep_time)
    return driver.page_source


with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

prefix = "https://www.finanzen.net/suchergebnis.asp?_search="
data = {d: data[d] for d in data if 'finanzen_net' not in data[d]}
names = [data[d]['Name'].replace(",", "") for d in data]
names = [n.split() for n in names]
names = [prefix + "+".join(n) for n in names]

ticker_names = [d for d in data]
ticker = [prefix + n for n in ticker_names]
ticker_names = ["finLink/" + n for n in ticker_names]

links = []
first = 0

for n, fi in zip(names, ticker_names):
    try:
        print(sp(get_questions_html(n, 0.75)).find('tbody'))
        na = sp(get_questions_html(n, 0.75)).find('tbody').find('a').get('href')
        print(na)
    except:
        continue
    with open(fi, "w") as f:
        if "nachricht/" in na:
            continue
        f.write(na)
