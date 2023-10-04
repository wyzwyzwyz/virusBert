from collections import defaultdict
from os import device_encoding
from numpy.core.numeric import zeros_like
from torch.nn.init import zeros_
import torch
import torch.nn as nn
from torch.optim import Adam, optimizer
from torch.utils.data import DataLoader

import tqdm
from ..model import BERTLM, BERTMM, BERT, BERTEncoder

from ..model.classifier import BaseClassifier, CNNClassifier,MLPClassifier

from .warmup import WarmUpLR

from sklearn.metrics import confusion_matrix
import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns


class TrainClassifier:
    """
    BERTTuning  make the fine tuning BERT model with one classifier.

    please check the details on README.md with simple example.

    """
    def __init__(self, bert_encoder: BERTEncoder,  train_dataloader: DataLoader = None, test_dataloader: DataLoader = None,**kvargs):
        """
        :param bert: BERT model which you want to train
        :param biClassifier:model with classifier
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]

        **kvargs
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and kvargs["with_cuda"]
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("\t use cuda:", cuda_condition)

        # This BERT model will be saved every epoch

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.virus_flag = kvargs["virus_flag"]

        if kvargs["model_name"] == "MLP":
            self.classifier = MLPClassifier(bert_encoder,**kvargs)
        elif kvargs["model_name"] == "CNN":
            self.classifier = CNNClassifier(bert_encoder,**kvargs)
        else:
            pass
        
        self.classifier = self.classifier.to(self.device)
        self.pooling_strategy = kvargs["pooling_strategy"]
 
        '''配置训练的学习器
        '''
        self.batch_size =  kvargs["cbatch_size"]
        self.input_dim = kvargs["input_dim"]

        self.lr = kvargs["clr"]
        self.betas = (kvargs["cadam_beta1"],kvargs["cadam_beta2"])
        self.weight_decay = kvargs["cadam_weight_decay"]

        self.nlabels = kvargs["nlabels"]

        self.epoch = kvargs["cepochs"]

        self.warmup_epochs = kvargs["cwarmup_batchs"]
        self.gradient_accum = kvargs["cgradient_accum"]

        self.lr_patience = kvargs["clr_patience"]
        self.lr_gamma = kvargs["clr_gamma"]
        self.lr_adjust_gamma = kvargs["clr_adjust_gamma"]

        self.checkpoint = kvargs["ccheckpoint"]
        self.sf = kvargs["logging_path"]

        if not kvargs["finetuning"]:
            for param in self.classifier.encoder.parameters():
                param.requires_grad = False

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT Finetuning" %
                  torch.cuda.device_count())
            
            self.classifier = nn.DataParallel(self.classifier)

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.classifier.parameters(), lr=self.lr,
                          betas=self.betas, weight_decay=self.weight_decay)

        # Setting the warm up LR  with hyper-param
        self.warmup_scheduler = WarmUpLR(optimizer=self.optim, warmup_epochs=self.warmup_epochs, gamma=self.lr_gamma,
                                         patience=self.lr_patience, lr_adjust_gamma=self.lr_adjust_gamma, verbose=True)

        # Using Binary cross entropy Loss function for predicting the class

        self.criterion = nn.BCEWithLogitsLoss()
        # self.criterion = nn.CrossEntropyLoss()
        # self.criterion = nn.BCELoss()

        if self.checkpoint != None:
            self.load_checkpoint(self.checkpoint)

        print("Total Parameters:", sum([p.nelement()
              for p in self.classifier.parameters()]))


    @torch.no_grad()
    def get_fixed_embed(self,data_loader,epoch):
        '''
        @msg:固定bert预训练参数，直接获取序列的嵌入
        @param: 
        @return:
        '''
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        for step,data in data_iter:
            [bert_inputs, bert_labels, segment_labels,
                is_next_labels, seq_names] = data

            bert_inputs = bert_inputs.cuda()

            # sequence_output 是每个word 的hidden 维嵌入
            pooled_output,sequence_output = self.encoder(bert_inputs,segment_info=None,output_all_encoded_layers=False,pooling_strategy=self.pooling_strategy)


        return pooled_output,sequence_output

    def embed_once(self,bert_inputs):
        '''
        @msg:微调bert预训练参数，获取序列的嵌入
        @param: 
        @return:
        '''
        # Setting the tqdm progress bar
        pooled_output,sequence_output = self.encoder(bert_inputs,segment_info=None,output_all_encoded_layers=False,pooling_strategy=self.pooling_strategy)
        return pooled_output,sequence_output

    def train(self, epoch):
        # self.model = self.model.train()
        self.iteration(self.train_data, epoch)

    @torch.no_grad()
    def test(self, epoch):
        self.iteration(self.test_data, epoch, train=False)

    def iteration(self, data_loader, epoch, input_type="cds",train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every peoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0

        if not self.virus_flag:
            for step,data in data_iter:
                [bert_inputs, bert_labels, segment_labels,
                    is_next_labels, seq_names] = data

                bert_inputs = bert_inputs.cuda()

                labels = torch.zeros((len(seq_names)))
                idx = 0
                for name in seq_names:
                    try:
                        theid = int(name.split("|")[1])
                    except TypeError:
                        print("Error label format!")
                        labels[idx] = 0
                    else:
                        labels[idx] = theid
                    idx += 1
                labels = labels.cuda()
                
                # 1.先获取序列的embed
                # pooled_output是每条cds 的 hidden*1 维嵌入 ；sequence_output 是每个word 的hidden 维嵌入

                if input_type=="cds":
                    #2.1 直接对cds 进行分类
                    # 二分类问题
                    pred_labels = self.classifier(bert_inputs,None,False,self.pooling_strategy)
                
                elif input_type == "multi-cds-add":
                    #2.2 将相同virus name 的cds embed相加
                    pass

                elif input_type == "multi-cds-concat":
                    #2.3 将相同 virus name 的cds embed 进行拼接
                    pass
                    
                else:
                    print("Invalid input type!")
                
                # 对labels进行独热编码
                onehot_labels = torch.eye(2)[labels.long(),:].cuda()
                loss = self.criterion(pred_labels,onehot_labels)

                acc =torch.eq(pred_labels.argmax(dim=-1),labels).sum()

                avg_loss += loss.item()
                
                # backward and optimization only in train
                if train:
                    loss /= self.gradient_accum
                    loss.backward()
                    if ((step + 1) % self.gradient_accum) == 0:
                        self.optim.step()
                        self.warmup_scheduler.loss_buffer = avg_loss / (step + 1)
                        self.warmup_scheduler.step(loss * self.gradient_accum)
                        self.optim.zero_grad()

                lr = self.optim.param_groups[0]['lr']

                if self.sf != None:
                    if str_code == "train":
                        with open(self.sf, 'a+') as f:
                            f.write("%s Epoch %d : Loss %2.6f : Hit %d : Avg Loss : %2.6f : lr : %2.9f \n" % (
                                str_code, epoch, loss.item()*self.gradient_accum, acc, avg_loss / (step+1), lr))
                    else:
                        with open(self.sf, 'a+') as f:
                            f.write("%s Epoch %d : Loss %2.6f : Hit %d : Avg Loss : %2.6f : lr : %2.9f \n" % (
                                str_code, epoch, loss.item(), acc, avg_loss / (step+1), lr))

                if str_code == "train":
                    data_iter.set_description(
                        f'{str_code.title()} Epoch {epoch} : Loss {loss.item() * self.gradient_accum: 2.6f} : Hit {acc} : Avg Loss {avg_loss/(step+1): 2.6f} : lr : {lr: 2.9f}')

                else:
                    data_iter.set_description(
                        f'{str_code.title()} Epoch {epoch} : Loss {loss.item() : 2.6f} : Hit {acc} : Avg Loss {avg_loss/(step+1): 2.6f} : lr : {lr: 2.9f}')


        else:
            pass
            # 先加载所有编码数据 
            # 再合并相同virus name 的数据作为分类的输入
        

    def cac_accuracy(self, labels, pred_labels):
        """计算分类准确率
        """

        pass

    def cac_precision(self, confusion_matrix):

        pass
    
    def drawROC(self,precision,recall,confusion_matrix):
        plt.figure(figsize=(16,16))
        sns.heatmap(confusion_matrix)
        plt.savefig("/workspace/MG-tools/bert-old-new/data/cm.png")

        plt.figure(figsize=(16,16))
        plt.plot(recall,precision)
        plt.savefig("/workspace/MG-tools/bert-old-new/data/roc.png")


    def cac_rp(self, confusion_matrix):
        """根据confusion矩阵计算recall 和precision
        """
        precision =[confusion_matrix[i][i]/np.sum(confusion_matrix[:][i]) for i in range(len(confusion_matrix))]
        recall = [confusion_matrix[i][i]/np.sum(confusion_matrix[i][:]) for i in range(len(confusion_matrix))]
        with open('/workspace/preprocessing-czj/output/cm', 'w+') as f:
            for i in confusion_matrix:
                for j in i:
                    f.write(str(j) + '\t')
                f.write('\n')
        avg_p = np.mean(precision)
        avg_r = np.mean(recall)
        
        return precision,recall,avg_p,avg_r

    def cac_each_class_index(self, data):
        accrucy = dict()
        precision = dict()
        recall = dict()
        for label, pred_label in data.items():
            cnt = 0
            for p in pred_label:
                if p == label:
                    cnt += 1
            accrucy[label] = cnt/len(pred_label)

    def cac_each_class_index(self,data):
        accrucy = dict()
        precision = dict()
        recall = dict()
        for label,pred_label in data.items():
            cnt = 0
            for p in pred_label:
                if p == label:
                    cnt +=1
            accrucy[label] = cnt/len(pred_label)
        

    def prediction(self, data_loader):
        k = 5

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="Predicting",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # p = [list() for _ in range(k)]
        p = list()
        l = list()
        labels_set = set()
        for step, dataset in data_iter:

            [bert_inputs, bert_labels, segment_labels,
                is_next_labels, seq_names] = dataset
            
            labels = torch.zeros((len(seq_names)))
            idx = 0
            for name in seq_names:
                try:
                    theid = int(name.split("|")[1])
                except TypeError:
                    print("Error label format!")
                    labels[idx] = 0
                    labels_set.add(0)
                else:
                    labels[idx] = theid
                    labels_set.add(theid)
                idx += 1

            bert_inputs = bert_inputs.cuda()
            labels = labels.cuda()  # shape:(batchsize)
            # shape:(batchsize,nlabels)
            pred_matrix = self.model(bert_inputs, None)

            # pred = torch.max(pred_matrix, 1)[1]
            pred = torch.sort(pred_matrix)[1][:, -5:]

            # 每个类对应的预测类别
            # p.extend(pred.cpu().numpy().tolist())
            pp = pred.detach().cpu().numpy()
            # for _ in range(k):
            #     p[_].extend(pp[_])
            # l.extend(labels.cpu().numpy().tolist())
            for _ in range(len(pred)):
                if int(labels[_]) in pp[_]:
                    p.extend([labels[_].cpu().numpy().tolist()])
                else:
                    p.extend([pp[_][0]])
            l.extend(labels.cpu().numpy().tolist())
        # 计算混淆矩阵
        l = np.array(l)
        p = np.array(p)
        cm = np.zeros((69, 69))
        # for _ in range(k):
        #     cm += confusion_matrix(l, p[_], labels = [_ for _ in range(69)])
        cm = confusion_matrix(l.astype(np.int), p, labels = [_ for _ in range(69)])
        precision,recall,avg_p,avg_r = self.cac_rp(cm)
        self.drawROC(precision,recall,cm)
        print("The avg precision :{} The avg recall:{}".format(avg_p,avg_r))
        if self.sf!=None:
            with open(self.sf,"a+") as f:
                f.write("".join(str(e) for e in l))
                f.write("".join(str(e) for e in p))
                f.write("\n")


    def save(self, epoch, path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path

        """
        state = {'model': self.classifier.state_dict(
        ), 'optimizer': self.optim.state_dict(), 'epoch': epoch}
        torch.save(state, path)
        self.classifier.to(self.device)
        # print(path)
        print("EP:%d Model Saved on:%s" % (epoch, path))

    def load_checkpoint(self, checkpoint_path):
        # self.model = nn.DataParallel(self.model)
        model_ckp = torch.load(checkpoint_path)
        self.classifier.load_state_dict(model_ckp['model'])
        self.epoch = model_ckp['epoch']
        print("Loading checkpoint!")
        self.optim.load_state_dict(model_ckp['optimizer'])
