'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-12 13:54:32
LastEditors: Yang jin
LastEditTime: 2021-10-13 04:38:06
'''

from tqdm import tqdm
import random
import os

def load_fasta(path):
    res = []
    with open(path,'r') as f:
        for line in f.readlines():
            if line.startswith(">"):
                ones = []
                virus_name = line.strip('\n').split("|")[-1]
                ones.append(virus_name)
            else:
                cds = line.strip("\n")
                ones.append(cds)
            if len(ones) == 2:
                res.append(ones)
    return res

def get_all_virus(genome_seq_list:list,load_path:str):
        # 获取所有的virus name
    all_virus_name = set()

    if load_path==None and genome_seq_list != None:
        for i,info in enumerate(tqdm(genome_seq_list)):
            virus_name, virus_lineage, host_lineage, cds = info
            all_virus_name.add(virus_name)
    else:
        genome_seq_list = load_fasta(load_path)
        for i,info in enumerate(tqdm(genome_seq_list)):
            virus_name, cds = info
            all_virus_name.add(virus_name)

    return list(all_virus_name)

def split_dataset(genome_seq_list:list,random_rate:list,random_state:int=0,load_path =None,save_path = "/workspace/datasets/cds/homo/"):
    '''
    @msg:  将数据按照random rate切分为training set validation set testing set
    @params:  
        genome_seq_list:[[virus name,virus lineage,host lineage,cds]]
        random_rate : [r1,r2]
        random_state: 随机种子
        load_path:if genome_seq_list = None,那么load_path不能为空
    @return:  
        tra：(msg\ncds)
        val：(msg\ncds)
        test：(msg\ncds)
    '''    
    random.seed(random_state)
    all_virus_name  = get_all_virus(genome_seq_list,load_path)
    count = len(all_virus_name)

    tra_,val_,test_ = [],[],[]
    source_list = list(range(0,count))

    random.shuffle(source_list)

    #tra,val,test
    pre = int(random_rate[0]*count)
    tra_ = source_list[:pre]
    aft = int((random_rate[0]+random_rate[1])*count)
    val_ = source_list[pre:aft]
    test_ = source_list[aft:]

    tra_virus = []
    val_virus = []
    test_virus = []

    for i in range(count):
        if i in tra_:
            tra_virus.append(all_virus_name[i])
        elif i in val_:
            val_virus.append(all_virus_name[i])
        else:
            test_virus.append(all_virus_name[i])

    tra_genome = []
    val_genome = []
    test_genome = []

    if load_path==None and genome_seq_list != None:
        for info in tqdm(genome_seq_list):
            virus_name, virus_lineage, host_lineage, cds = info
            if virus_name in tra_virus:
                tra_genome.append(info)
            elif virus_name in val_virus:
                val_genome.append(info)
            else:
                test_genome.append(info)
    else:
        genome_seq_list = load_fasta(load_path)
        for info in tqdm(genome_seq_list):
            virus_name, cds = info
            if virus_name in tra_virus:
                tra_genome.append(info)
            elif virus_name in val_virus:
                val_genome.append(info)
            else:
                test_genome.append(info)

    if save_path:
        with open(os.path.join(save_path,"tra.fna"),'w') as filehandle:
            for seq_ in tqdm(tra_genome):
                filehandle.write(">label|1|"+seq_[0]+"\n")
                filehandle.write(seq_[-1]+"\n")

        with open(os.path.join(save_path,"val.fna"),'w') as filehandle:
            for seq_ in tqdm(val_genome):
                filehandle.write(">label|1|"+seq_[0]+"\n")
                filehandle.write(seq_[-1]+"\n")
        
        with open(os.path.join(save_path,"test.fna"),'w') as filehandle:
            for seq_ in tqdm(test_genome):
                filehandle.write(">label|1|"+seq_[0]+"\n")
                filehandle.write(seq_[-1]+"\n")

    return tra_genome,val_genome,test_genome
      
tra_genome,val_genome,test_genome = split_dataset(None,[0.6,0.2],100,load_path="/workspace/datasets/cds/Euk/non-euk/non-eukaryota.cds.fna",save_path="/workspace/datasets/cds/Euk/non-euk/")