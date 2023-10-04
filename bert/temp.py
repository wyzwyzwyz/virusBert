import pickle
import numpy as np
import torch


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
    data_result={}

    for i in range(608):
        if(i>0):
            break
        path1 = "./embedding_path1/frag1_embed" + str(i) + ".pk"
        path2 = "./embedding_path1/frag2_embed" + str(i) + ".pk"
        data1 = load_pickle(path1)
        data2 = load_pickle(path2)
        data2=np.array(data2.cpu())
        print(data2[0])
        for j in range(256):
            if(data1[j] in data_result):
                data_result[data1[j]] = np.concatenate((data_result[data1[j]], data2[j]), axis=0)
            else:
                data_result[data1[j]]=data2[j]


    # for i,data in enumerate(data_result.values()):
    #     print(data)







