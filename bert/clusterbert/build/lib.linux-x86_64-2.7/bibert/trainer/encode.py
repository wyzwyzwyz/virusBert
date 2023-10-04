'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-27 09:21:38
LastEditors: Yang jin
LastEditTime: 2022-03-11 13:07:12
'''

import torch
import torch.nn as nn

from ..model import BERTEncoder

import pickle
from tqdm import tqdm
from collections import defaultdict
import os


def save_dict(datadict,path):
    # 将数据转移到cpu上
    for k,v in datadict.items():
        datadict[k] = v.cpu().numpy()
    with open(path +".pk", 'wb') as f:
        pickle.dump(datadict, f)


class SeqEmbedding:
    def __init__(self, bert_encoder: BERTEncoder, hidden,  embed_path:str=None,with_cuda=True):
        super(SeqEmbedding).__init__()

        self.bert_encoder = bert_encoder

        self.hidden = hidden

        self.embed_path = embed_path

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("\t use cuda:", cuda_condition)
        self.bert_encoder.eval()

        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.bert_encoder = nn.DataParallel(
                self.bert_encoder.to(self.device))

    def encode_once(self, inputs, segment_labels, output_all_encoded_layers, pooling_strategy):
        return self.bert_encoder(inputs, segment_labels, output_all_encoded_layers, pooling_strategy)

    @torch.no_grad()
    def get_encodes(self, data_loader, output_all_encoded_layers, pooling_strategy):
        '''
        @msg:
            获取每条输入序列经过pooling_strategy（CLS，MAX）后的1*hidden嵌入
        @param:
        @return:
            字典类型：
                {frag_name:embed}
            frag_name格式：
                >index|d|label|src|contig_name
        '''
        str_code = "Encode"

        data_iter = tqdm(enumerate(data_loader),
                              desc="EP_%s:" % (str_code),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        datadict = defaultdict()

        for nbatch,dataset in data_iter:
            [bert_inputs, bert_labels, segment_labels,
                      seq_names] = dataset

            bert_inputs = bert_inputs.cuda()
            bert_labels = bert_labels.cuda()
            segment_labels = segment_labels.cuda()  # here segment_labels is zero matrix

            pooled_output,_ = self.encode_once(
                bert_inputs, segment_labels, output_all_encoded_layers, pooling_strategy)

            del _ # 每个kmer的嵌入

            if  len(seq_names) != pooled_output.shape[0]:
                print("\n----There are no {}th file 's sequence length is {} which is not matched ----".format(nbatch,len(bert_inputs)))
                print("\t The seq names are:",seq_names)
                print("\t----The seq names ' length is {}----".format(len(seq_names)))
                print("\t----The embed length is {}----".format(len(pooled_output)))
                continue

            for seq_name,vec in zip(seq_names,pooled_output): # 考虑存储问题，数据量越多，所需内存和GPU存储越多，会有存储不足问题
                datadict[seq_name] = vec   
            
            #直接写入硬盘
            if self.embed_path:
                save_dict(datadict,os.path.join(self.embed_path,"frag_embed"+str(nbatch)))
                datadict.clear()
            
        return datadict 

    


    





