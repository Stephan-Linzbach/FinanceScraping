import json

with open("./Europe_Usa_cleaned_profil.json") as f:
    data = json.load(f)
#%%
def clean(k):
    foundkey = None
    for yy in y:
        if yy in k:
            foundkey = yy
    if not foundkey:
        return k
    new = k.find(foundkey) 
    return k[new:]
     
#%%
#%%

y = ['Personal', 'Aktion\u00e4rsstruktur', 'Bilanz']
for d in data:
    x = data[d]
    keys = list(x.keys())
    values = list(x.values())
    rel_keys = set()
    keys = [clean(k) for k in keys]
    data[d] = {k : v for k, v in zip(keys, values)}.copy()

#%%
len('Bilanz (in ')
#%%

for d in data:
    x = data[d]
    keys = list(x.keys())
    values = list(x.values())
    for k in keys:
        if 'Bilanz' in k:
            order = k[11:14]
            currency = k[16:19]
            index = keys.index(k)
            keys[index] = k[:6] + k[20:]
    keys += ['Order', 'Currency']
    values += [order, currency]
    data[d] = {k: v for k, v in zip(keys, values)}

with open("./Europe_Usa.json_cleaned_final.json", "w") as f:
    json.dump(data, f)
#%%

currencies = set()

for d in data:
    currencies.add(data[d]['Currency'])

currencies
#%%

counts = {c : 0 for c in currencies}

#%%

for d in data:
    counts[data[d]['Currency']] += 1

counts
#%%
sum(counts.values())
#%%

wrong = [d for d in data if data[d]['Currency'] == ') -']
        
#%%
for w in wrong:
    print(data[w])
    break

