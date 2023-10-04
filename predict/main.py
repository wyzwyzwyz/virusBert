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

if __name__ == '__main__':
    # list_data=load_pickle("./train_data.pk")
    # print(len(list_data.keys()))
    # for i,data in enumerate(list_data.keys()):
    #     print(data)


    data_result = {}
    # data_result = defaultdict(list)

    for i in range(608):
        if (i%10==0):
            bb = time.time()  # 结束时间
            cc = bb - aa
            print(str(i)+":用了"+str(cc//1)+"秒！")

        path1 = "./embedding_path1/frag1_embed" + str(i) + ".pk"
        path2 = "./embedding_path1/frag2_embed" + str(i) + ".pk"
        data1 = load_pickle(path1)
        data2 = load_pickle(path2)
        data2 = np.array(data2.cpu())
        for j in range(256):
            if (data1[j] in data_result):
                data_result[data1[j]] = np.concatenate((data_result[data1[j]], data2[j]), axis=0)
            else:
                data_result[data1[j]] = data2[j]

    for i, data in enumerate(data_result.keys()):
        if((int)(data_result[data].shape[0])/768<10):
            data_result.pop(data)
        else:
            data_result[data]=data_result[data][:7680]

    save_pickle(data_result, "./train_data.pk")

    # for i, data in enumerate(data_result.keys()):
    #     print(data[6])
    #     print((int)(data_result[data].shape[0])/768)















    # with open("./label_do_and_do_not-host-all.corpus", 'rb') as f:
    #     data_list=f.readlines()
    #     data_map={}
    #     sum1=0
    #     sum2=0
    #     sum3=0
    #     print("总长度为："+str(len(data_list)))
    #     for i,data in enumerate(data_list):
    #         if(i>139790):
    #             continue
    #         temp=(len(data)-9)/7//500
    #         if(temp<10):
    #             sum1+=1
    #         elif(temp<20):
    #             sum2+=1
    #         else:
    #             sum3+=1
    #
    #
    #     print(sum1)
    #     print(sum2)
    #     print(sum3)


    # with open("./label_do-not-host-all.corpus", 'rb') as f:
    #     data_list=f.readlines()
    #     print(len(data_list))



