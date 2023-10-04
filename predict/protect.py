import torch
import time
from model import DNN_model
from dataset import MyData
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import optim
# 使用的参数
epoch_num=100000
batchsize_num=16
learnRate=0.001
device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")

path_train="./data_train.pk"
path_validation="./data_test.pk"
# path_test="./data_test.pk"

lablepath_train="./lable_train.pk"
lablepath_validation="./lable_test.pk"

dataset_train=MyData(path_train,lablepath_train)
dataset_validation=MyData(path_validation,lablepath_validation)
# dataset_test=MyData(path_test)

dataloader_train=DataLoader(dataset=dataset_train,batch_size=batchsize_num,shuffle=True)
dataloader_validation=DataLoader(dataset=dataset_validation,batch_size=batchsize_num,shuffle=True)
# dataloader_test=DataLoader(dataset=dataset_test,batch_size=batchsize_num,shuffle=True)

def make_batch(inputs):
    x,y=inputs
    list_y=[]
    for i in range(len(y)):
        # print(int(y[i][0]))
        if(int(y[i][0])==0):
            list_y.append([0])
        else:
            list_y.append([1])
    return (torch.tensor(x,device=device),torch.tensor(list_y).view(len(y)))

model=DNN_model(768)
model.to(device)
model.load_state_dict(torch.load("./model_wehave/state_dict_model_100.pth"))
# loss_function=F.nll_loss()
optimizer =  optim.SGD(model.parameters(),lr=learnRate)

print("开始训练！")
aa = time.time()  # 开始时间

for epoch in range(epoch_num):

    corret_num = 0
    for i, data in enumerate(dataloader_validation):
        model.eval()
        input, label = make_batch(data)  # 数据加载
        output = model(input)  # 3.计算前馈
        for j in range(output.shape[0]):
            output_label = output[j]
            print("dcdscd")
            print((int)(label[j].item()))
            print(int(torch.max(F.softmax(output_label).unsqueeze(0), 1).indices.item()))
            if ((int)(label[j].item()) == int(torch.max(F.softmax(output_label).unsqueeze(0), 1).indices.item())):
                corret_num += 1




