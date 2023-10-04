import pickle
import numpy as np
import torch
from collections import defaultdict
import time

aa = time.time()  # 开始时间




def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data
# data_list=load_pickle("./train_data.pk")
# print(len(data_list.keys()))
# aa = time.time()  # 开始时间
# for i in range(10000):
#     temp=load_pickle("lable_test.pk")
#
# bb = time.time()
# cc = bb - aa
# print("用了" + str(cc // 1) + "秒！")
# aa = time.time()  # 开始时间
# for i in range(10000):
#     temp=load_pickle("lable_train.pk")
#
# bb = time.time()
# cc = bb - aa
# print("用了" + str(cc // 1) + "秒！")


aa = time.time()
temp=load_pickle("./NewclassDistance1000.pk")
bb = time.time()
cc = bb - aa
print("用了" + str(cc // 1) + "秒！")
aa = time.time()
for i in range(1):
    for j,data in enumerate(temp):
        if(data==("1","2")):continue
bb = time.time()
cc = bb - aa
print("用了" + str(cc // 1) + "秒！")

