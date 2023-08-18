import json
from news_scraper import *

with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

succ = daily_fetch()

print(succ)
