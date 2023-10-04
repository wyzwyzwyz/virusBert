'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-29 03:43:12
LastEditors: Yang jin
LastEditTime: 2021-10-29 04:02:48
'''
from os import O_NOCTTY
from numpy.lib.npyio import save
from tqdm import tqdm
from collections import Counter
import pandas as pd
import xlwt

def describe_otu(path:str,save_path:str=None):
    otus = []
    with open(path,"r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            otus.append(line.strip().split("\t")[1])
    
    each_otu_cnt = Counter(otus)
    # otus = {}
    # for k,v in each_otu_cnt:
    #     otus[k] = v
    
    if save_path:
        writer = pd.ExcelWriter(save_path)
        df = pd.DataFrame()
        df["OTU"] = list(each_otu_cnt[:][0])
        df["#Contigs"] = list(each_otu_cnt[:][1])
        df.to_excel(writer)
        writer.save()
        writer.close()
    return each_otu_cnt


result = describe_otu("/data1/bert-data/vamb-data/skin/reference.tsv","/data1/bert-data/vamb-data/skin/otu.xls")

    
