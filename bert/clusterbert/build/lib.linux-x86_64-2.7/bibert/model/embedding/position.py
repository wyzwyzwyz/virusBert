'''
Author: CAI Zhijie
Date: 2021-10-27 02:26:51
LastEditors: CAI Zhijie
LastEditTime: 2021-10-27 02:43:20
Description: 
FilePath: /BiBERT/bibert/model/embedding/position.py
'''
import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = True

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float().exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).cuda()
        self.register_buffer('pe', pe)
        # self.pe = pe

    def forward(self, x):
        return self.pe[:, :x.size(1)]
