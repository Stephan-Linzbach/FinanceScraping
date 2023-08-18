import json
from yahoo_scraper import *

with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

succ = scraper(data, 'MINUTE')

print(succ)
