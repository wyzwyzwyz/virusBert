'''
Description: 
Version: 
Author: Yang jin
Date: 2022-03-10 09:31:59
LastEditors: Please set LastEditors
LastEditTime: 2022-03-15 13:27:54
'''

import os 
from matplotlib import  pyplot as plt
import numpy as np

def parser_loss(file):
    loss_ ,avg_loss_,lr_ = [],[],[]

    with open(file,"r") as f :
        for line in f.readlines():
            _ = line.strip().split(":")
            # assert len(_) == 6
            loss_ = float(_[1].strip("Loss "),".4f")
            avg_loss_ = float(_[3].strip(),".4f")
            lr_ = float(_[-1])

    return loss_,avg_loss_,lr_

def plot_loss(dir,filename):
    ''' loss格式:
        train Epoch id : Loss 数值：Avg Loss : 数值 ：lr : 数值
    '''
    file = os.path.join(dir,filename)
    loss_,avg_loss_,lr_ = parser_loss(file)
    x = np.arange(len(loss_))
    plt.plot(x,loss_,markers="*")
    plt.plot(x,avg_loss_,markers="-")
    plt.plot(x,lr_,markers="o")
    
dir = "/workspace/vamb-data/1229fa/"
filename = "corpus3/model_params_bs_8sl_512/0214-3mer-0.0001.log"
plot_loss(dir,filename)




 


    




