import json
from finanzen_net_scraper import *

with open("Europe_Usa_cleaned.json") as f:
    data = json.load(f)

succ = getKursZiele(data)

print(succ)
