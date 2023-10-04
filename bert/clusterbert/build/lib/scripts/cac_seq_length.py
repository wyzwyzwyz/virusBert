'''
Author: CAI Zhijie
Date: 2021-10-13 07:00:27
LastEditors: Yang jin
LastEditTime: 2021-10-25 22:37:02
Description: Genaralize metadata of the sequences
FilePath: /bien-torch/kit/length_stat.py
'''
#%%
from inspect import FullArgSpec
from  tqdm import tqdm
import seaborn as sns
import os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(data_path:str,filename:str="contigs.fna"):
    """
    [[msg, seq],
     [msg, seq],
     ...
     [msg, seq]]
    """
    def del_None_msg(genome_seq):

        new_genome_seq = []
        for i, one in enumerate(genome_seq):
            if one[0].strip()=="":
                continue


            new_genome_seq.append(one)

        return new_genome_seq

    genome_seq_list = []

    with open(os.path.join(data_path , filename), 'r') as f:
        print('----load genome file with not None----')
        one_msg = []
        seq_msg = ''

        for line in tqdm(f.readlines()):

            if line[0] == '>':

                if seq_msg != '':
                    one_msg.append(seq_msg)  # 若当前seq_msg不为空，则就是已读取完完整序列
                    genome_seq_list.append(one_msg)

                    seq_msg = ''  # 为下次序列做准备
                    one_msg = []

                id_msg = line.strip('\n')  # 除去首尾回车符号

                # collection
                one_msg.append(id_msg)
            else:
                seq_msg += line.strip('\n') #序列拼接即可

        # 上面循环无法收集最后一条数据
        one_msg.append(seq_msg)
        genome_seq_list.append(one_msg)

    print('there are', len(genome_seq_list), 'uncleaned genome seqs')
    genome_seq_list = del_None_msg(genome_seq_list)
    print('and', len(genome_seq_list), 'cleaned  genome seqs !')

    return genome_seq_list

#%%
def fna2fasta(data_path:str=None,filename:str ='contigs.fna',genome_seq_list:list=None):
    '''
    @msg: 将fna格式（一行表述，多行基因）转换为标准的fasta格式，即一行描述，一行基因
    @param:
        laod_data return list like:
            [msg,seq]
            ...
            [msg,seq]
    @return:
        file like:
        msg\nseq\n....msg\nseq
    '''
    if genome_seq_list ==None:
        genome_seq_list = load_data(data_path,filename)
    
    name = "".join(filename.split('.')[:-1])+".fa"
    with open(os.path.join(data_path,name),"w") as filehandle:
        for msg,seq in tqdm(genome_seq_list):
            filehandle.write(msg+'\n')
            filehandle.write(seq+'\n')

    return genome_seq_list

    
def label_data(path:str,filename:str,label:str):
    '''
    @msg: 给不同来源的每条序列标记上其来源
    @param:
        labels:list 序列的来源
        fasta或 fna文件：>msg\nseq
    @return: 
        >label|src|msg\nseq
    '''
    _ = os.path.join(path,filename)
    with open(_,"r") as filehandle:
        lines = filehandle.readlines()
        for j,line in tqdm(enumerate(lines)):
            if line.startswith(">"):
                lines[j] = ">"+"label|"+label+"|"+line.lstrip(">")

    with open(_,"w") as filehandle:
        filehandle.writelines(lines)
#%%
path = "/data1/bert-data/vamb-data"
names = ["airways","gi","metahit","oral","skin","urog"]
for n in names:
    file = os.path.join(path,n)
    label_data(file,"contigs.fa",n)
    
#%%
def length_stat(path: str=None,genome_list:list=None):
    '''
    @msg:获取fasta序列的基因碱基数量
    @param: 
        path ,genome_list
        only one is valid
    '''
    len_list = []

    if path!=None and genome_list == None:
        with open(path,'r') as f:
            for line in tqdm(f.readlines()):
                if line.startswith(">"):
                    continue
                else:
                    len_list.append(len(line.strip('\n')))
    else:
        for msg,seq in tqdm(genome_list):
            len_list.append(len(seq))
                
    return len_list
#%%
path = "/data1/bert-data/vamb-data/skin"
genome_seq_list = load_data(path,"contigs.fna")
_ = fna2fasta(data_path = path,genome_seq_list=genome_seq_list)
len_list = length_stat(genome_list=_)
pd.DataFrame(len_list).describe()
#%%
len_list_skin = len_list
#%%
len(air_len_list)
#%%
def plot_len_dis(data):
    sns.lineplot(x="contig size",y = "number", data=data,ci=None)
#%%
from collections import Counter
air_dict = Counter(air_len_list)
air = sorted(air_dict.items(),key = lambda x:x[0])
data = pd.DataFrame(air,columns=["contig size","number"])
data
#%%
plot_len_dis(data)
# %%
