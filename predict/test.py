import pickle
import numpy as np
import torch
import re
def save_pickle(data, file_name):
    f = open(file_name, "wb")
    pickle.dump(data, f)
    f.close()

def load_pickle(file_name):
    f = open(file_name, "rb+")
    data = pickle.load(f)
    f.close()
    return data

list1=[]
list2=[]
list3=[]


data_list=load_pickle("./train_data.pk")
my_set1=set()
my_set2=set()
# print(len(data_list.keys()))


temp=data_list.keys()
for i, data in enumerate(list(temp)):
        if((int)(data_list[data].shape[0])/768<10):
            data_list.pop(data)
        else:
            data_list[data]=data_list[data][:7680]


for i,data in enumerate(data_list.keys()):
    lable = int(str(data[6]))
    temp = data[:data.rfind('|')]
    temp = temp[temp.rfind('|') + 1:]
    if(lable==0):
        my_set1.add((temp))
    else:
        my_set2.add((temp))

my_set=my_set1.intersection(my_set2)
my_set.remove("")
my_set.remove("Homo sapiens")
print(my_set)
print(len(my_set))
for i,datai in enumerate(my_set):
    sum=0
    num1=0
    num2=0
    print(datai)
    for j,dataj in enumerate(data_list.keys()):
        temp = dataj[:dataj.rfind('|')]
        temp = temp[temp.rfind('|') + 1:]
        if(temp==datai):
            # print(temp)
            sum+=1
            if(int(str(dataj[6]))==0):
                num1+=1
            else:
                num2+=1
    print("总数为："+str(sum)+"能感染人的数量为："+str(num1)+"不能感染人的数量为："+str(num2))






# resoult_tensor=torch.zeros(1,7680)
#
# for i,datai in enumerate(my_set):
#     print(i)
#     for j,dataj in enumerate(data_list.keys()):
#         if(j%1000==0):
#             print(j)
#         temp = dataj[:dataj.rfind('|')]
#         temp = temp[temp.rfind('|') + 1:]
#         if(temp==datai):
#             lable = int(str(dataj[6]))
#             temp_tensor = torch.FloatTensor(data_list[dataj][:7680])
#             temp_tensor[0] = lable
#             temp_tensor = temp_tensor.unsqueeze(0)
#             resoult_tensor = torch.cat([resoult_tensor, temp_tensor], 0)
#
# resoult_tensor=resoult_tensor[1:]
# print(resoult_tensor.shape)
# save_pickle(resoult_tensor,"./data_hard.pk")




#
# sum_num=len(data_list)
# train_num=int((sum_num*0.4)//1)
# validation_num=int((sum_num*0.6)//1)
#
#
resoult_tensor=torch.zeros(1,7680)

for i,data in enumerate(data_list.keys()):
    if(i%1000==0):
        print(i)
    temp = data[:data.rfind('|')]
    temp = temp[temp.rfind('|') + 1:]
    if(temp not in my_set):
        lable = int(str(data[6]))
        temp_tensor = torch.FloatTensor(data_list[data][:7680])
        temp_tensor[0] = lable
        temp_tensor = temp_tensor.unsqueeze(0)
        resoult_tensor = torch.cat([resoult_tensor, temp_tensor], 0)




resoult_tensor=resoult_tensor[1:]
save_pickle(resoult_tensor,"./data_train.pk")


#
# data_list1=torch.cat([resoult_tensor[:train_num], resoult_tensor[validation_num:]], 0)
# data_list2=resoult_tensor[train_num:validation_num]
# # data_list3=resoult_tensor[validation_num:]
#
# print(resoult_tensor.shape)
# print(data_list1.shape)
# print(data_list2.shape)
# # print(data_list3.shape)

# save_pickle(data_list1,"./data_train.pk")
# save_pickle(data_list2,"./data_validation.pk")

# save_pickle(data_list3,"./data_test.pk")








