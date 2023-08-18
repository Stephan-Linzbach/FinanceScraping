import sys
import json
from yahoo_finance_api2 import share
from yahoo_finance_api2.exceptions import YahooFinanceError
from datetime import date
import os

settings = {'YEAR': {'period': share.PERIOD_TYPE_YEAR,
                     'history': 1,
                     'freq': share.FREQUENCY_TYPE_DAY},
            'HOUR': {'period': share.PERIOD_TYPE_YEAR,
                     'history': 2,
                     'freq': share.FREQUENCY_TYPE_HOUR},
            'MINUTE': {'period': share.PERIOD_TYPE_DAY,
                       'history': 7,
                       'freq': share.FREQUENCY_TYPE_MINUTE}, }


def scraper(equity_list, period):
    """
        Gathers data for every equity from equity_list
    """
    if period not in settings.keys():
        print("Period must be in settings.keys(): " + str(settings.keys()))
        return False

    if 'Symbol' not in equity_list[0]:
        print("Equity list objects need key 'Symbol'")
        return False

    symbols = [s for s in equity_list]

    my_share = [share.Share(s) for s in symbols]

    symbol_data = {s.symbol: retrieve(s, period) for s in my_share}

    today = date.today()

    return write_json_to(symbol_data, "yahoo_" + period + "_" + today.strftime("%Y/%m/%d"))


def retrieve(my_share, period):
    try:
        symbol_data = my_share.get_historical(settings[period]['period'],
                                              settings[period]['history'],
                                              settings[period]['freq'],
                                              1)
    except YahooFinanceError as e:
        print(e.message)
        return
    except KeyError as k:
        print(k)
        return
    return symbol_data


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
