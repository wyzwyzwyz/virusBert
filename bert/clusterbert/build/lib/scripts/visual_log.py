'''
Author: CAI Zhijie
Date: 2021-10-26 07:40:25
LastEditors: CAI Zhijie
LastEditTime: 2021-10-27 02:06:14
Description: 
FilePath: /BiBERT/scripts/visual_log.py
'''

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tqdm

mpl.rcParams['agg.path.chunksize'] = 20000

def visual_log(path: str, o_path: str, code: str='train', start=None, end=None):
    f = open(path, 'r')
    fig = plt.figure()
    l = list()
    al = list()
    lr = list()
    e = list()
    pbar = tqdm.tqdm(total=os.path.getsize(path))
    for line in f:
        if line.startswith('train'):
            t = line.strip('\n').split(':')
            # t1, t2, t3, t4 = float(t[1]), float(t[3]), float(t[5]), float(t[7])
            t1, t2, t3, t4 = int(t[0][-2]), float(t[1].split(' ')[2]), float(t[3].replace(' ', '')), float(t[-1])
            e.append(t1), l.append(t2), al.append(t3), lr.append(t4)
        pbar.update(len(line))

    d1 = pd.DataFrame.from_dict({'loss': l, 'avg_loss': al, 'epoch': e})
    d2 = pd.DataFrame.from_dict({'lr': lr, 'epoch': e})

    
    if end is not None:
        assert type(end) is int
        d1, d2 = d1[:end], d2[:end]
    if start is not None:
        assert type(start) is int
        d1, d2 = d1[start:], d2[start:]

    sns.set_style('darkgrid')
    fig, ax1 = plt.subplots(figsize=(20, 12))
    sns.lineplot(ax=ax1, data=d1, x=list(range(len(d1))), y='loss', hue='epoch', palette='pastel')
    sns.lineplot(ax=ax1, data=d1, x=list(range(len(d1))), y='avg_loss', hue='epoch', palette='dark')
    # ax1.lines[0].set_linestyle('--')
    for _ in range(e[-1]+2,len(ax1.lines)):
        ax1.lines[_].set_linestyle('--')
    ax2 = ax1.twinx()
    sns.lineplot(ax=ax2, data=d2, x=list(range(len(d1))), y='lr', hue='epoch')
    fig.savefig(o_path, dpi=400)

pp = '/workspace/datasets/vamb-data/tra/model_params_bs_8sl_512/10262223.log'
visual_log(pp, pp+'.png')
