import re
from datetime import date
import time
import requests
import json
import random
import os
from bs4 import BeautifulSoup as bs


def write_json_to(jsonList, name):
    try:
        name = "./scraped/" + name
        os.makedirs(name[:-7])
        with open(name, "w") as f:
            json.dump(jsonList, f)
    except Exception as e:
        with open(name, "w") as f:
            f.write(e)
        return False
    return True


def orderdByDate(expectations):
    orderdByDate = {}
    for k, v in expectations.items():
        date = {}
        for vv in v['expectations']:
            if date.get(vv['date'], False):
                date[vv['date']].append({'dir': vv['direction'], 'tar': vv['target']}.copy())
            else:
                date[vv['date']] = [{'dir': vv['direction'], 'tar': vv['target']}.copy()]
        orderdByDate[k] = date.copy()
    return orderdByDate


def getKursZiele(unternehmen):
    expectations = {}
    for n in unternehmen:
        try:
            page = requests.get("https://www.finanzen.net/kursziele/" + unternehmen[n]['finanzen.net']).text
            page = bs(page)
            table = page.find('table', {'class': 'table table-vertical-center'})
            analysis = []
            for t in table.findAll('tr'):
                row = t.findAll('td')
                if not row: continue
                print(row)
                try:
                    if row[1].div['class'][0] == 'backgroundYellowWhite':
                        direction = 0
                    if row[1].div['class'][0] == 'backgroundGreenWhite':
                        direction = 1
                    if row[1].div['class'][0] == 'backgroundRedWhite':
                        direction = -1
                    analysis.append({'analyser': row[0].text, 'target':
                        row[2].text, 'date': row[4].text,
                                     'direction': direction}.copy())
                except:
                    pass
            print(analysis)
            expectations[n] = {'link': unternehmen[n]['finanzen.net'], 'expectations': analysis}.copy()
        except:
            pass
        today = date.today()
        time.sleep(random.random())
    return write_json_to(expectations, "finanzen_net_kursziele_" + today.strftime("%Y/%m/%d"))


"""
def getDax(pages):
    unternehmen = {}
    for page in pages:
        page = bs(page)
        table = page.find('table', {'class' : 'table table-small table-hover'})
        for t in table.findAll('tr'):
            row = t.findAll('td')
            try:
                unternehmen[row[0].text.replace("\t", "").replace("\n","").replace("\r"," ").split()[0]] = row[0].find('a')['href']
            except:
                pass
        time.sleep(random.randint(2, 20)) 
    with open("dax.json", "w") as f:
        json.dump(unternehmen, f)

def avg(l):
    print(l)
    return sum(l)/len(l)

getDax([requests.get('https://www.finanzen.net/index/dax/30-werte').text,
           requests.get('https://www.finanzen.net/index/mdax/werte').text,
           requests.get('https://www.finanzen.net/index/sdax/werte').text,
           requests.get('https://www.finanzen.net/index/tecdax/werte').text,
           requests.get('https://www.finanzen.net/index/dow_jones/werte').text,
           requests.get('https://www.finanzen.net/index/cac_40/werte').text,
           requests.get('https://www.finanzen.net/index/nikkei_225/werte').text])

with open("dax.json") as f:
    unternehmen = getKursZiele(json.load(f))

with open("expectations.json", "w") as f:
    json.dump(unternehmen, f)

with open("expectations.json") as f:
    pD = orderdByDate(json.load(f))

with open("perDate.json", "w") as f:
    json.dump(pD, f)

with open("perDate.json") as f:
    for k, v in json.load(f).items():
        print("xxxxxxxxxxxxxxxxxxxx")
        print(k)
        smallest = ['test',  0]
        for kk, vv in v.items():
            print("--------------------------")
            print(kk)
            print(vv)
            dire = [vvv['dir'] for vvv in vv]
            print(sum(dire))
            if smallest[1] > sum(dire):
                smallest[0] = kk
                smallest[1] = sum(dire)
            print(len(dire))
            try:
                print(avg([int(re.sub("\D", "" , vvv['tar']))/100 for vvv in vv]))
            except:
                pass
        print(smallest)
        print("xxxxxxxxxxxxxxxxxxxx")
"""
