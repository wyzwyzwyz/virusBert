from torch.utils.data import Dataset
import pickle

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data
class MyData(Dataset):

    def __init__(self, path_data,path_lable):
        self.datalist = load_pickle(path_data)
        self.lablelist = load_pickle(path_lable)


    def __getitem__(self, idx):

        result=self.datalist[idx]
        lable=self.lablelist[idx]
        # print(result.shape)
        # print(lable)

        return result,lable

    def __len__(self):
        return len(self.datalist)
# class MyData(Dataset):
#
#     def __init__(self, path,num=0):
#         self.path = path
#         self.lenNum=num
#
#
#     def __getitem__(self, idx):
#         path = self.path + "/train_data.6kmer." + str(idx).zfill(8)
#         result =load_pickle(path)
#         lable=[]
#         for i in range(result.shape[0]):
#             print(i)
#             print(int(result[i][0]))
#             lable.append(str(int(result[i][0])))
#             print(lable)
#
#
#         # result=self.datalist[idx]
#         # lable=self.lablelist[idx]
#
#         return result,lable
#
#     def __len__(self):
#         return self.lenNum
