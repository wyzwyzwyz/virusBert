'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-15 03:39:52
LastEditors: Yang jin
LastEditTime: 2021-10-15 03:43:29
'''
#%%
import pickle

def loac_vocab(path):
    with open(path,"rb") as filehandle:
        try :
            return  pickle.load(filehandle)
        except EOFError:
            return None

vocab = loac_vocab("/workspace/MG-tools/BiBERT/data/cds.vocab")
# %%
