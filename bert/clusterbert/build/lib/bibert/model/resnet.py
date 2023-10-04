'''
Author: your name
Date: 2021-09-15 13:20:11
LastEditTime: 2021-10-19 07:19:58
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch-old/bert_pytorch/model/resnet.py
'''
from .bert import BERTEncoder
from torch import nn

class ResBasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_channels, out_channels,stride =1 ,downsample = None, **kvargs):
        super(ResBasicBlock, self).__init__()
        self.downsample = downsample

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 3, stride, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(self.out_channels)
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,3, 1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(self.out_channels)
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    
    def forward(self, input):
        residual = input
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample:
            residual = self.downsample(residual)
        x += residual
        x = self.relu(x)
        return x


class ResNetBuilder(nn.Module):
    def __init__(self,block, **kvargs):
        super(ResNetBuilder, self).__init__()
        self.nlabels = kvargs["nlabels"]

        self.dropout = nn.Dropout(kvargs["dropout"])

        self.seq_len = kvargs["seq_len"]
        self.in_channels = kvargs["in_channels"]
   
        self.layers_hidden = kvargs["layers_hidden"] # num_layer = [1024, 512, 256]
   
        self.kernel_size = kvargs["kernel_size"] # 3,3,3,3
        self.stride = kvargs["stride"] # 1 1 1 2
        self.padding = kvargs["padding"] # 1 1 1 3

        self.classifier = nn.ModuleList()

        self.conv1 = nn.Conv2d(self.in_channels, self.layers_hidden[0], kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)
        self.bn1 = nn.BatchNorm2d(self.layers_hidden[0])
        self.maxpool = nn.MaxPool2d(self.kernel_size, stride = self.stride, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, self.layers_hidden[0], self.layers_hidden[1], 1)
        self.layer2 = self._make_layer(block, self.layers_hidden[1], self.layers_hidden[1], 2)
        self.layer3 = self._make_layer(block, self.layers_hidden[1], self.layers_hidden[2], 1)

        self.avgpool = nn.AvgPool2d(kernel_size=(7, 1), stride=1)

        self.fc = nn.Linear(62976, 1024, bias=True) # 计算最后所得的维度
        self.fc2 = nn.Linear(1024, self.nlabels, bias=True) 

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        
    def _make_layer(self, block, in_channels, out_channels, num_block, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels*block.expansion, 
                    1, 
                    stride=stride, bias=True
                ),
                nn.BatchNorm2d(
                    out_channels*block.expansion
                )
            )
        layers = []
        layers.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(1, num_block):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)
        
    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.fc2(x)
        return x







