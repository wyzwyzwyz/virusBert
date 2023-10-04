'''
Descripttion: 
version: 
Author: Yang jin
Date: 2021-09-15 13:09:26
LastEditors: Yang jin
LastEditTime: 2021-10-18 08:35:18
'''
from .bert import BERTEncoder
from .config import CNNConfig
from torch import nn

class BERTCNN(nn.Module):
    """Supervised fine-tuning with labeled sequences using CNNText(2014 Yoon Kim)
    """
    pass
    # def __init__(self, bert_encoder: BERTEncoder, clConfig: CNNConfig, num_labels: int):
    #     super(BERTCNN, self).__init__()
    #     self.bert_encoder = bert_encoder
    #     self.num_labels = num_labels

    #     self.seq_len = clConfig.seq_len 
    #     self.in_dim = clConfig.hidden # 输入数据的维度nbatchsize*nkmers*hidden) 512
    #     self.out_channels = clConfig.out_channels  # 100
    #     self.batch_size = clConfig.batch_size
    #     self.kernel_size =clConfig.kernel_size

    #     self.pooled_size = clConfig.pooled_size

    #     self.out_dim = int(self.out_channels[-1]*(self.seq_len-self.kernel_size+1)/self.pooled_size)

    #     """model structure
    #     """
    #     self.conv1 = nn.Conv2d(
    #             in_channels=1, out_channels= self.out_channels[0], kernel_size=(self.kernel_size,clConfig.hidden))
    #     # self.conv2 = nn.Conv1d(in_channels=self.out_channels[0],out_channels=self.out_channels[1],kernel_size=6)
    #     self.relu = nn.ReLU()
    #     self.maxpool = nn.MaxPool2d((self.pooled_size,1))
        
    #     self.linear1 = nn.Linear(self.out_dim,num_labels)

    #     self.softmax = nn.LogSoftmax(dim=-1)

    # def forward(self, input_ids, token_type_ids=None, output_all_encoded_layers=False):

    #     seq_output = self.bert_encoder(
    #         input_ids, token_type_ids, output_all_encoded_layers)

    #     output = seq_output.unsqueeze(1) #shape (batchsize,1,seq_len,hidden)

    #     out = self.relu(self.conv1(output))

    #     out = self.maxpool(out)

    #     # out = self.relu(self.maxpool(self.conv2(out)))
    #     out = out.view(out.shape[0],-1)
      
    #     output_labels = self.softmax(self.linear1(out))

    #     return  output_labels