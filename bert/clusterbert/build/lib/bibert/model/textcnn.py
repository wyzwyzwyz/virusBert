'''
Author: your name
Date: 2021-09-16 13:25:04
LastEditTime: 2021-10-18 08:36:07
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch-old/bert_pytorch/model/textcnn.py
'''
from .bert import BERTEncoder
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F

class textCNN(nn.Module):
    def __init__(self,  num_labels: int):
        super(textCNN, self).__init__()
        pass

    #     self.in_channels = clConfig.hidden
    #     self.out_channels = clConfig.out_channels
    #     self.bert_encoder = bert_encoder
    #     ks = [3, 6, 9, 12, 15]
    #     knum = [2, 2, 2, 2, 2]
    #     self.convs = nn.ModuleList([nn.Conv1d(self.in_channels, knum[_], ks[_]) for _ in range(len(ks))])
    #     self.dropout = nn.Dropout(0.5)
    #     self.fc = nn.Linear(np.sum(knum), num_labels)
    #     self.relu = nn.ReLU()
    #     self.softmax = nn.LogSoftmax(dim=-1)

    # def forward(self, input_ids, token_type_ids=None, output_all_encoded_layers=False):
    #     pooled_output, seq_output = self.bert_encoder(
    #         input_ids, token_type_ids, output_all_encoded_layers)
    #     output = seq_output.permute(0, 2, 1)
    #     x = [self.relu(conv(output)) for conv in self.convs]
    #     x = [F.max_pool1d(line, line.size(-1)) for line in x]

    #     x = torch.cat(x, 1)
    #     x = self.dropout(x.squeeze())
    #     x = self.softmax(self.fc(x))
    #     return x
        
