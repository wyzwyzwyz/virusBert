'''
Author: your name
Date: 2021-09-15 14:09:22
LastEditTime: 2021-10-27 03:23:45
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch-old/bert_pytorch/model/config.py
'''
class BERTConfig(object):
    def __init__(self,vocab_size,hidden,n_layers,attn_heads,cluster_hidden=768,dropout=0.1,hidden_dropout_prob=0 ):
        super().__init__()
        self.vocab_size=vocab_size
        self.hidden=hidden
        self.n_layers=n_layers
        self.attn_heads=attn_heads
        self.dropout = dropout
        self.hidden_dropout_prob = hidden_dropout_prob
        
        