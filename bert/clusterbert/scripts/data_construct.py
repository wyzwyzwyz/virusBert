'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-11 02:32:50
LastEditors: Yang jin
LastEditTime: 2021-10-25 08:08:09
'''
#%%
from collections import defaultdict,Counter
from enum import EnumMeta
from numpy.core.fromnumeric import resize
from tqdm import tqdm
import json
import os
import pandas as pd
import random

#%%
def load_data(data_path:str,filename:str="contig.fna"):
    """
    [[msg, seq],
     [msg, seq],
     ...
     [msg, seq]]
    """
    def del_None_msg(genome_seq):

        new_genome_seq = []
        for i, one in enumerate(genome_seq):
            # get all host lineages
            _ = one[0].split('|')
            virus_name,virus_lingeages,host_lineages  = _[0],_[3],_[4]
            
            if virus_name == '':
                continue

            if host_lineages=='':
                continue
                
            if virus_lingeages=='':
                continue

            new_genome_seq.append(one)

        return new_genome_seq

    genome_seq_list = []

    with open(os.path.join(data_path , filename), 'r') as f:
        print('----load genome file with not None----')

        lines = f.readlines()

        one_msg = []
        seq_msg = ''

        for line in tqdm(lines):

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

    print('there are', len(genome_seq_list), 'uncleaned virus cds of genome seqs')
    genome_seq_list = del_None_msg(genome_seq_list)
    print('and', len(genome_seq_list), 'cleaned virus cds of genome seqs !')

    return genome_seq_list


#%%
def deal_same_name(hosts_seqs, host_flag=True):
    """
    find same name host in one lineage, and mark them different

    e.g.
        Anopheles, Anopheles_1, Anopheles_2
    """

    print('----deal same name----')
    new_hosts_seq = []
    same_name_set = set()

    for k, one in enumerate(tqdm(hosts_seqs)):
        if host_flag:
            lineage = one[2]    # host lineage
        else:
            lineage = one[1]    # virus lineage

        members = lineage.split(';')

        # find same name host's index
        index = [j for j, x in enumerate(members) if members.count(x) > 1]

        # change its name
        for i, x in enumerate(index):
            if i > 0:
                # the second same host to be host_1
                members[x] += '_' + str(i)
                same_name_set.add(members[x])

        new_lineage = ';'.join(members)     # restore lineage
        if host_flag:
            new_hosts_seq.append((one[0], one[1], new_lineage, one[3]))
        else:
            new_hosts_seq.append((one[0], new_lineage, one[2], one[3]))

    print('same name num =', len(same_name_set), same_name_set)

    return new_hosts_seq

def load_virus_host_msg(genome_seq_list):
    """
    split information for the whole virus

    [[msg, seq],
     [msg, seq],
     ...
     [msg, seq]]

    :param genome_seq_list: all cds genome seq
    :return new_genome_seq_: [[virus name, virus lineage, host lineage, cds genome]]
    :return list(virus_msg_set_): (virus name, virus lineage, host lineage)
    :return cds_num_list_: (virus name, cds) distribution
    """
    def get_host_msg(seq_host_msg_):
        seq_list_ = seq_host_msg_.split('|')
        virus_name_ = ' '.join(seq_list_[0].split(' ')[1:])

        virus_lineage_ = seq_list_[3].split('; ')
        all_virus_lineage_ = ';'.join(virus_lineage_)
        host_lineage_ = seq_list_[4].split('; ')
        all_host_lineage_ = ';'.join(host_lineage_)
        return virus_name_, all_virus_lineage_, all_host_lineage_

    print('----load virus & host msg----')
    virus_msg_set_ = set()
    virus_name_list_ = []
    new_genome_seq_ = []

    for ps in genome_seq_list:
        msg = get_host_msg(ps[0])   # get msg of one cds
        virus_msg_set_.add(msg)     # while msg
        virus_name_list_.append(msg[0]) # only virus name

        # [virus name, virus lineage, host lineage, cds genome]
        new_genome_seq_.append(list(msg)+[ps[1]])   # [ps[1]] means cds genome sequence

    # 计算各个病毒的CDS个数 (存在个别CDS识别host，与同病毒的其他CDS不同，识别host lineage不够完整or该段CDS就是无法识别完整host)
    # cds片段不是连接在一起，应该按病毒名字统计
    cds_num_list_ = list(Counter(virus_name_list_).items())     # [(virus name: cds num)]
    assert len(cds_num_list_) == len(set(virus_name_list_))

    print('there are total {} virus seqs,{} different virus seqs !'.format(len(virus_name_list_), len(set(virus_name_list_))))
    return deal_same_name(new_genome_seq_), list(virus_msg_set_), cds_num_list_

#%%
def get_host_level(virus_msg_list:list,save_flag=False,save_path="/workspace/MG-tools/virHost/data/"):
    '''
    @msg: 
        获取每个taxonomy级别的host
    @params:  
        [(virus name, virus lineage, host lineage)]
    @return: 
        {
            level 0:[host names]
            level 1:[host names]
            ...
        }
    '''    
    
    all_level_host_dict = defaultdict(set)

    for msg in tqdm(virus_msg_list):
        assert len(msg) == 3
        host_lingeage =  msg[2].split(";")
        for level,name in enumerate(host_lingeage):
            # if "Viruses"  in name or "virus" in name:
            #     print(level,":",msg)
            #     break
            all_level_host_dict[level].add(name) # 提取出各个taxonomy等级的host
    
    print("there are {} taxonomy host".format(len(all_level_host_dict)))
    for k_,v_ in all_level_host_dict.items():
        print("\t there are {} hosts in taxonomy level {}".format(len(v_),k_))
        all_level_host_dict[k_] = list(v_)

    if save_flag:
        with open(os.path.join(save_path , 'level_host_name.json'),'w') as f:
            f.write(json.dumps(all_level_host_dict,ensure_ascii=False, indent=4, separators=(',', ':')))

    return all_level_host_dict

#%%
def get_key_cds(new_genome_seq_list:list,host_key:str,host_flag=True,save_path="/workspace/MG-tools/virHost/data/"):
    '''
    @msg: 获取指定host_key的[msg,cds]
    @params:  
        new_genome_seq_list:[[virus name, virus lineage, host lineage, cds genome]]
        host_key:key1,key2... 
            eg 
                Human,non-Human,(Homo:人属，仅存homo sapiens智人一种物种，其余都是non-Human)
                Eukaryota,non-Eukaryota,（真核生物）
                Metazoa,non-Metazoa,(后生动物亚界，动物界除原生动物门以外的所有多细胞动物门类的总称)
                Chordata,non-Chordata,（脊索动物门，是动物界最高等的一门，包括头索，尾索，脊椎动物亚门）
                Mammalia,non-Mammalia （哺乳动物纲，脊椎动物亚门的一纲。通称兽类。身体被毛；体温恒定；胎生）
            
    @return:  
        key1_file(msg\ncds)
        key2_file(msg\ncds)
        ...
            
        key1:{virus/host species 1:num,virus/host  species 2:num...}(virus lineage 最后一个字段是virus species)
        key2:{virus/host species 1:num,virus/host  species 2:num...}
        ...
    '''    
    host_key_seqs = []
    non_host_key_seqs = []
    unknown_seqs = []

    host_virus_sp_dict = defaultdict(set)
    non_host_virus_sp_dict = defaultdict(set)
    unknow_host_virus_sp_dict = defaultdict(set)

    print("----split seqs into host or non-host or unknow host----")

    for i,info in enumerate(tqdm(new_genome_seq_list)):
        virus_name,virus_lineage,host_lineage,cds = info
        
        if host_flag:
            members = [e.lower() for e in host_lineage.split(";")]
        else:
            members = [e.lower() for e in virus_lineage.split(";")]
        
        virus_members = virus_lineage.split(";")

        if members.count(host_key) > 0 :
            # 搜索所有host_lineage中包含homo的序列
            host_key_seqs.append([virus_name,virus_members[-1],cds])
            host_virus_sp_dict[virus_members[-1]].add(virus_name)
            
        else:
            # 不包含homo的序列不一定是non-human的序列，有可能是Hominoidea(人猿总科）Hominidae(人科)Homininae（人亚科）中缺失属的分类
            if host_key == "homo":
                if members[-1] in ["hominoidea","hominidae","homininae"] or  virus_members[-1]  in host_virus_sp_dict:
                    # 存在相同的virus ，可能由于信息缺失，没有分类到species，无法确定是否可以感染人类，
                    # 这些virus 也不能出现在non-host数据集中，会影响准确率
                    unknown_seqs.append([virus_name,virus_members[-1],cds])
                    unknow_host_virus_sp_dict[virus_members[-1]].add(virus_name)
                else:
                    non_host_key_seqs.append([virus_name,virus_members[-1],cds])
                    non_host_virus_sp_dict[virus_members[-1]].add(virus_name)

            non_host_key_seqs.append([virus_name,virus_members[-1],cds])
            non_host_virus_sp_dict[virus_members[-1]].add(virus_name)


    clean_non_host_virus_sp_dict = defaultdict(set)

    for key_,value_ in non_host_virus_sp_dict.items():
        if key_ in host_virus_sp_dict:
            for e in value_:
                unknow_host_virus_sp_dict[key_].add(e)
        else :
            clean_non_host_virus_sp_dict[key_] = value_
            

    def get_dict_length(d):
        res ={}
        for key,value in d.items():
            res[key] = len(value)
        return res

    unknow_host_virus_sp_dict = get_dict_length(unknow_host_virus_sp_dict)
    clean_non_host_virus_sp_dict = get_dict_length(clean_non_host_virus_sp_dict)
    host_virus_sp_dict = get_dict_length(host_virus_sp_dict)

    if save_path:
        with open(os.path.join(save_path,host_key+".cds.fna"),"w") as host_file:
            print("----save host seq----")
            for seq_ in tqdm(host_key_seqs):
                host_file.write(">label|0|"+seq_[0]+"\n")
                host_file.write(seq_[-1]+"\n")

        with open(os.path.join(save_path,"non-"+host_key+".cds.fna"),"w") as non_host_file:
            print("----save non host seqs----")
            for seq_ in tqdm(non_host_key_seqs):
                # 从non-host 中剔除掉virus members[-1] 已经出现在host file 中的所有序列，加入到unknown_seqs中  
                if seq_[1] in host_virus_sp_dict:
                    unknown_seqs.append(seq_)
                    unknow_host_virus_sp_dict[seq_[1]] += 1
                else:
                    non_host_file.write(">label|1|"+seq_[0]+"\n")
                    non_host_file.write(seq_[-1]+"\n")

        result_excel1 = pd.DataFrame({host_key:list(host_virus_sp_dict.keys()),"samples":list(host_virus_sp_dict.values())})
        result_excel2 = pd.DataFrame({"non-"+host_key:list(clean_non_host_virus_sp_dict.keys()),"samples":list(clean_non_host_virus_sp_dict.values())})
        result_excel3 = pd.DataFrame({"unknown":list(unknow_host_virus_sp_dict.keys()),"samples":list(unknow_host_virus_sp_dict.values())})

        with pd.ExcelWriter(os.path.join(save_path,host_key+"_binary-virus_species.xls")) as writer:
            result_excel1.to_excel(writer,sheet_name=host_key,header = [host_key+"virus species","#samples"],index=False)
            result_excel2.to_excel(writer,sheet_name="non-"+host_key,header = ["non-"+host_key+"virus species","#samples"],index=False)
            result_excel3.to_excel(writer,sheet_name="unknow host",header = ["unknown host virus species","#samples"],index=False)
        
        host_file.close()
        non_host_file.close()
    
    return host_key_seqs,non_host_key_seqs,unknown_seqs

#%%
def describe_virus_species(new_genome_seq_list:list,host_flag=False,save_path="/workspace/MG-tools/virHost/data/"):
    '''
    @msg: 按照virus的病毒分类或者宿主分类，获取每个类目下所有的病毒数目
    @params:  
        virus_msg_list :[[virus name, virus lineage, host lineage,cds]]
    @return:  
        {
            virus/host species 1:virus nums,
            virus/host species 2:virus nums,(not cds)
            ...
        }
    '''
    virus_sp_dict = defaultdict(set)
    virus_num_dict = {}

    for k, one in enumerate(tqdm(new_genome_seq_list)):
        if host_flag:
            lineage = one[2]    # host lineage
        else:
            lineage = one[1]    # virus lineage

        members = lineage.split(';')[-1] # species 是倒数第一个字段吗？病毒分类而言是的，宿主不一定

        virus_sp_dict[members].add(one[0])
    
    for k_,v_ in virus_sp_dict.items():
        virus_num_dict[k_]= len(v_)
    
    virus_sp_list = list(virus_num_dict.keys())
    virus_num_list = list(virus_num_dict.values())

    result_excel = pd.DataFrame()
    result_excel["virus species"] = virus_sp_list
    result_excel["#samples"] = virus_num_list

    if save_path:
        result_excel.to_excel(os.path.join(save_path,"virus_species_num.xls"),header=["virus species","#samples"])

    return virus_sp_dict

def describe_virus_name(genome_seq_list:list,fasta_flag=True,save_path=None):
    all_virus_name = defaultdict(int)

    for i,info in enumerate(tqdm(genome_seq_list)):
        if fasta_flag:
            virus_name, cds = info
        else: 
            virus_name, virus_lineage, host_lineage, cds = info
        all_virus_name[virus_name] += 1
    
    # all_virus_name = list(all_virus_name)
    if save_path:
        result = pd.DataFrame(all_virus_name)
        with open(save_path,"w") as writer:
            result.to_excel(writer,)

    return list(all_virus_name.keys()),list(all_virus_name.values())

#%%
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

def random_sample(genome_seq_list:list,random_count:int,random_state:int=0,load_path =None,save_path = "/workspace/datasets/cds/homo/" ):
    '''
    @msg:  根据homo_binary-virus_species.xls内容，随机抽取1144条virus samples(相同virus name为同一个病毒，为一个sample)
    @params:  
        genome_seq_list:[[virus name, virus lineage, host lineage, cds genome]]（需要随机抽取的数据)
        random_count:抽取数量
        random_state:随机种子
    @return:  
        sampled_non_homo.cds.fna(msg\ncds)
        [[virus name, virus lineage, host lineage, cds genome]]
    '''   
    # 获取所有的virus name
    random.seed(random_state)

    all_virus_name = get_all_virus(genome_seq_list,load_path)
    
    remained_ = random.sample(len(all_virus_name),random_count)

    remained_virus_name = [all_virus_name[i] for i in remained_]
    
    remained_genome = []

    if load_path==None and genome_seq_list != None:
        for info in tqdm(genome_seq_list):
            virus_name, virus_lineage, host_lineage, cds = info
            if virus_name in remained_virus_name:
                remained_genome.append(info)
    else:
        genome_seq_list = load_fasta(load_path)
        for info in tqdm(genome_seq_list):
            virus_name, cds = info
            if virus_name in remained_virus_name:
                remained_genome.append(info)
        

    if save_path:
        with open(save_path,'w') as filehandle:
            for seq_ in tqdm(remained_genome):
                filehandle.write(">label|1|"+seq_[0]+"\n")
                filehandle.write(seq_[-1]+"\n")

    return remained_genome,remained_virus_name

#%%
data_path = "/workspace/MG-tools/virHost/data/"
genomes= load_data(data_path)
new_genome_seq, virus_msg_list,cds_dist = load_virus_host_msg(genomes)

# #%%
# all_level_host_dict = get_host_level(virus_msg_list,True)
# virus_sp_dict = describe_virus_species(new_genome_seq)

#%%
host_key_seqs,non_host_key_seqs,unknown_seqs = get_key_cds(new_genome_seq,"eukaryota")

#%%
remained_genome,remained_virus_name = random_sample(None,228,load_path="/workspace/datasets/cds/homo/remained_non-homo.cds.fna.shf",save_path="/workspace/datasets/cds/homo/tra_non-homo.cds.fna")

# remained_genome,remained_virus_name = random_sample(None,660,load_path="/workspace/datasets/cds/homo/remained_non-homo.cds.fna.shf",save_path="/workspace/datasets/cds/homo/tra_non-homo.cds.fna")
#%%
seqs = load_fasta("/workspace/datasets/cds/homo/non-human/test.fna")
host_virus_name = describe_virus_name(seqs)
len(host_virus_name)
#%%

if __name__=="__main__":
    data_path = "/workspace/MG-tools/virHost/data/"
    genomes= load_data(data_path)
    new_genome_seq, virus_msg_list,cds_dist = load_virus_host_msg(genomes)
    host_key_seqs,non_host_key_seqs,unknown_seqs, host_virus_sp_set,non_host_virus_sp_set,unknow_host_virus_name_list = get_key_cds(new_genome_seq,"homo")





    


