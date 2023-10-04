'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-27 09:21:38
LastEditors: Yang jin
LastEditTime: 2021-10-29 08:44:44
'''
import torch
import torch.nn as nn

from tqdm import tqdm
from collections import defaultdict
import pickle

from ..model import BERTEncoder

class BERTEmbedding:
    def __init__(self, bert_encoder: BERTEncoder, hidden,   with_cuda=True):
        super(BERTEmbedding).__init__()

        self.bert_encoder = bert_encoder

        self.hidden = hidden


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
                     is_next_labels, seq_names] = dataset

            bert_inputs = bert_inputs.cuda()
            bert_labels = bert_labels.cuda()
            segment_labels = segment_labels.cuda()  # here segment_labels is zero matrix
            is_next_labels = is_next_labels.cuda()

            pooled_output,_ = self.encode_once(
                bert_inputs, segment_labels, output_all_encoded_layers, pooling_strategy)

            del _ # 每个kmer的嵌入

            assert len(seq_names) == pooled_output.shape[0]

            for seq_name,vec in zip(seq_names,pooled_output):
                datadict[seq_name] = vec  

        return datadict

    
class Cluster:
    def __init__(self,data):
        super(Cluster).__init__()
        self.data = data
    
    def data_preprocess(self):
        '''
        @msg: 
            先利用已预训练好的模型对数据进行kmer嵌入，获取每个Kmer的嵌入向量，再Kmer的嵌入向量->fragment的嵌入向量->Contig的嵌入向量。
            Kmer嵌入向量（kmers*hidden)-> Fragment嵌入向量(1*hidden)
                由BERTPooler类实现，有多种pooling策略：MAX,CLS,MEAN
            （self.data 为Fragment嵌入向量）
            Fragment嵌入向量-> Contig嵌入向量（关键）
                需要先获取所有的Fragment嵌入向量，再通过拼接策略，拼接为Contig的嵌入向量
                拼接策略：
                    1. 横向拓展策略[Fragment1 Fragment2 ....Fragmentm padding] (1*d)
                        按照Fragment在切割前的顺序做横向拼接，拼接成固定维度d的一维向量，需要确定合适的向量维度d 以至于平衡舍弃的Fragment和padding
                    2. 先纵向拓展再自乘A = [[Fragment1],[Fragment2],...,[Fragmentm]] R = A^T*A(hidden*hidden)
                        按照Fragment在切割前的顺序做纵向拼接，无需padding，需要设计合理的聚类算法对矩阵进行聚类。
                        存在时空消耗过大的问题
                    3. 纵向拓展策略[[Fragment1],[Fragment2],...,[Fragmentm],padding] (d*hidden)
                        按照Fragment在切割前的顺序做纵向拼接，拼接成d*hidden的矩阵，需要确定d,以及设计合理的聚类算法对矩阵进行聚类。
                        存在时空消耗过大的问题
        @param:
            self.data 格式：
                {
                    frag_name:embed
                }
        @return:
            contig_data:
                {
                    contig_name:embed
                }
        '''

    def kmeans(self):
        pass

    def iter_cluster(self):
        pass

    





