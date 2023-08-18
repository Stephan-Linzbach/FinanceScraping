import json
import time
from selenium.webdriver.chrome.options import Options
import glob
from bs4 import BeautifulSoup as bs
import os
import requests
import selenium 

#%% get html
def get_questions_html(url, sleep_time):
    print( 'Loading questions page...')
    chromeOptions = Options()
    chromeOptions.headless = True
    driver = selenium.webdriver.Chrome('/home/stephan/chromedriver', options=chromeOptions)
    try:
        driver.set_page_load_timeout(10)
        driver.get(url)
    except Exception as e:
        chromeOptions = Options()
        driver = selenium.webdriver.Chrome('/home/stephan/chromedriver')
        driver.set_page_load_timeout(10)
        try:
            driver.get(url)
        except:
            pass
    try:
        if not driver.page_source:
            print("No Source")
            return None
    except:
       return None 
    """
    for i in range(1, times_to_scroll):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(sleep_time)
    """
    time.sleep(sleep_time)
    return driver.page_source
#%%
def do_it(l):
    return list(set(l))
#%%
with open("./Europe_Usa.json_cleaned_final.json") as f:
     data = json.load(f)
#%%
names = [s[12:] for s in glob.glob("./out_links/*")]

data = {k : v for k, v in data.items() if k not in names}
#%%

for i, d in enumerate(data):
    if 'Adresse' not in list(data[d].keys()):
        with open("./out_links/" + d, "w") as f:
            json.dump({}, f)
        continue
    if 'Internet' not in list(data[d]['Adresse'].keys()):
        with open("./out_links/" + d, "w") as f:
            json.dump({}, f)
        continue
    print(i)
    url = data[d]['Adresse']['Internet'][0]
    print(url)
    if url == '':
        continue
    text = get_questions_html(url, 0)
    if not text:
        continue
    a = [a['href'] for a in bs(text).find_all('a', href=True)]
    twitter = do_it([t for t in a if 'twitter' in t])
    facebook = do_it([t for t in a if 'facebook' in t])
    youtube = do_it([t for t in a if 'youtube' in t])
    linkedin = do_it([t for t in a if 'linkedin' in t])
    all_known = twitter + facebook + youtube + linkedin
    all_extern = [t for t in a if 'https' in t and not t in all_known]
    for k in all_known:
        print(k)
    links = {}
    if len(twitter):
        links['Twitter'] = twitter
    if len(facebook):
        links['Twitter'] = facebook
    if len(youtube):
        links['Twitter'] = youtube
    if len(linkedin):
        links['Twitter'] = linkedin
    if len(all_extern):
        links['Extern'] = all_extern
    with open("./out_links/" + d, "w") as f:
        json.dump(links, f)

