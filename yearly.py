import json
from yahoo_scraper import *

with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

succ = scraper(data, 'HOUR')
succ = scraper(data, 'YEAR')

print(succ)
