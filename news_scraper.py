import gdelt
from newsfetch.news import newspaper
from datetime import date, timedelta
import json
import os


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


def daily_fetch():
    try:
        gd = gdelt.gdelt()
        today = date.today() - timedelta(days=25)
        gkg = json.loads(gd.Search([today.strftime("%d/%m/%Y")], table='gkg', output='json'))
        gkg = [d for d in gkg if d['V2Themes']]
        gkg_econ = [d for i, d in enumerate(gkg) if 'ECON' in gkg[i]['V2Themes']]
        succ = write_json_to(gkg_econ, "gkg_" + today.strftime("%Y/%m/%d") + ".json")
        articles = {}
        for g in gkg_econ:
            news = newspaper(g['DocumentIdentifier'])
            articles[g['GKGRECORDID']] = news.get_dict
            # articles[g['GKG']] = {'headline' : news.headline,
            #                      'article' : news.article,
            #                      'keywords' : news.keywords,
            #                      'publication' : news.publication,
            #                      'author' : news.author,
            #                      'summary' : news.summary,
            #                      'section' : news.section}
        succ_art = write_json_to(articles, "articles_" + today.strftime("%Y/%m/%d") + ".json")
    except Exception as e:
        print(e)
        return False
    return succ_art


daily_fetch()
