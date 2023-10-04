'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-12 13:51:43
LastEditors: Yang jin
LastEditTime: 2021-10-26 03:31:12
'''

import tqdm
import os
import random


def fa_shuffle(path, o_path):
    l = list()
    f = open(path, 'r')
    pbar = tqdm.tqdm(total=os.path.getsize(path))
    for line in f:
        if line.startswith('>'):
            l.append([line])
        else:
            l[-1].extend([line])
        pbar.update(len(line))
    print('read complete, start shuffling.\n')
    random.shuffle(l)
    print('shuffling complete, start output.\n')
    g = open(o_path, 'w+')
    for i in tqdm.tqdm(l):
        for j in i:
            g.write(j)
    print('output complete.\n')


fa_shuffle('/workspace/datasets/vamb-data/val.fa.500.frag.6mer', '/workspace/datasets/vamb-data/val.fa.500.frag.6mer.shf')