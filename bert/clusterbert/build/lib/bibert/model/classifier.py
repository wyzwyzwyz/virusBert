'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-17 06:44:33
LastEditors: Yang jin
LastEditTime: 2021-10-19 03:33:49
'''
from .bert import BERTEncoder
from .resnet import ResNetBuilder,ResBasicBlock

from torch import nn

from abc import abstractmethod

from torch import Tensor

class BaseClassifier(nn.Module):
    def __init__(self,bert_encoder: BERTEncoder,**kvargs):
        super(BaseClassifier,self).__init__()
        self.encoder = bert_encoder
    
    @abstractmethod
    def forward(self, input_ids,segment_info,output_all_encoded_layers=False,pooling_strategy="MAX",virus_flag = False)->Tensor:
        pass

class MLPClassifier(BaseClassifier):
    def __init__(self,  bert_encoder: BERTEncoder,**kvargs):
        super(MLPClassifier, self).__init__(bert_encoder,**kvargs)
    
        self.dropout = nn.Dropout(kvargs["dropout"])

        self.classifier = nn.ModuleList()

        in_dim = kvargs["input_dim"]
        self.layers_hidden = kvargs["layers_hidden"]
        self.nlabels = kvargs["nlabels"]

        for i in range(len(self.layers_hidden)):
            out_dim = self.layers_hidden[i]
            self.classifier.append(nn.Linear(in_dim, out_dim))
            self.classifier.append(nn.BatchNorm1d(out_dim))
            self.classifier.append(nn.ReLU())
            self.classifier.append(self.dropout)
            in_dim = out_dim

        self.classifier.append(
            nn.Linear(self.layers_hidden[-1], self.nlabels))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids,segment_info,output_all_encoded_layers=False,pooling_strategy="MAX",virus_flag = False):
        if virus_flag:
            pass
        else:
            output,seqence_embed = self.encoder(input_ids,segment_info,output_all_encoded_layers,pooling_strategy)

        for layer in self.classifier:
            output = layer(output)

        output_labels = self.softmax(output)

        return  output_labels


class CNNClassifier(BaseClassifier):
    def __init__(self,  bert_encoder: BERTEncoder,**kvargs):
        super(CNNClassifier, self).__init__(bert_encoder,**kvargs)

        self.classifier = ResNetBuilder(block=ResBasicBlock,**kvargs)
        self.softmax = nn.Softmax(dim=-1)

    
    def forward(self, input_ids,segment_info,output_all_encoded_layers=False,pooling_strategy="MAX",virus_flag = False):
        if virus_flag:
            pass
        else:
            out,seqence_embed = self.encoder(input_ids,segment_info,output_all_encoded_layers,pooling_strategy)
            output = seqence_embed.permute(0, 2, 1).unsqueeze(3)

        output_labels = self.softmax(self.classifier(output))

        return  output_labels
