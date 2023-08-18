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
    driver.get(url)
    """
    for i in range(1, times_to_scroll):
        driver.execute_script('window.scrollTo(0, document.body.scrollHeight)')
        time.sleep(sleep_time)
    """
    time.sleep(sleep_time)
    return driver.page_source

#%% Load Data

with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

#%%

p_a = path +"/UltraInstinctMoneyPrinting/profil/"
len(p_a)

#%% Clean data

data = {d : data[d] for d in data if 'finanzen_net' in data[d]}

already = [d[48:] for d in glob.glob(p_a+"*")]

data = {k :  v for k, v in data.items() if k not in already}

#%% Get Pages
len(data)
#%% Get Pages

prefix = "https://www.finanzen.net/unternehmensprofil/"

names = [data[d]['finanzen_net'].replace('-aktie','') for d in data]

#%% Symbols

symbol = [d for d in data]

#%% Show Names

names

#%% Urls

url = [prefix+n for n in names]

#pages = [requests.get(u).text for u in url]

#%%

text = [get_questions_html(u, 0) for u in url]

#%%

for u, s in zip(url, symbol):
    text = get_questions_html(u, 0)
    boxes = bs(text).find_all('div', {'class' : 'box'})


    right_boxes = [b for b in boxes if b.find('h2', {'class' : 'box-headline'})]
    right_n_boxes = [b.find('h2', {'class' : 'box-headline'}).text for b in right_boxes]

    r_b = {n : b.find('tbody') for n, b in zip(right_n_boxes, right_boxes)}



    results = {}

    for b in r_b:
        try:
            tr = r_b[b].find_all('tr')
        except:
            print(r_b[b])
            continue
        td = [t.find_all('td') for t in tr]
        if 'Aktionärsstruktur' in b or 'Bilanz' in b or 'Personal' in b:
            results[b] = {tdd[0].text : [t.text for t in tdd[1:]] for tdd in td}.copy()
        if 'Management' in b:
            try:
                text = [str(t).split('<br/>') for t in td]
                entries = [(e[1].replace('</td>]', ''), e[0].replace('[<td>', '')) for e in text]
                results[b] = {pos : names for pos, names in entries}
            except:
                pass
        if 'Adresse' in b:
            new_add = {}
            new_add['address'] = td[0][0].text
            new_add.update({tdd[0].text : [t.text for t in tdd[1:]] for tdd in td[1:]}.copy())
            results[b] = new_add
    with open(path + "/UltraInstinctMoneyPrinting/profil/" + s, "w") as f:
        print(results)
        json.dump(results, f)


#%% Integrating results 

already = {d[48:] : d for d in glob.glob(p_a+"*")}


with open(path + "/UltraInstinctMoneyPrinting/Europe_Usa_cleaned.json") as f:
    data = json.load(f)

for d in data:
    try:
        with open(already[d]) as f:
            alr = json.load(f)
            data[d].update(alr)
    except:
        print(d)
        continue
#%%
data

#%%

with open(path + "/UltraInstinctMoneyPrinting/Europe_Usa_cleaned_profil.json", "w") as f:
    json.dump(data, f)

#%%

unkown = [d for d in data if "Adresse" not in data[d]] 
len(unkown )

