'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-26 12:02:11
LastEditors: Yang jin
LastEditTime: 2021-10-26 13:56:12
'''
#%%
import pickle 

path = "/workspace/datasets/vamb-data/1026.vocab"
with open(path,"rb") as f:
    voc = pickle.load(f)
# %%
freq = list(voc.freqs.values())
# %%
import pandas as pd
cnts = pd.DataFrame(freq)
cnts.describe()
# %%
total = 0
total_cnt= 0
for e in freq:
    if e >= 100:
        total += 1
        total_cnt += e

# %%