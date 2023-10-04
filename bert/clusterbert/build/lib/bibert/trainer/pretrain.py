'''
Author: CAI Zhijie
Date: 2021-09-14 03:36:34
LastEditTime: 2021-10-13 12:49:37
LastEditors: Yang jin
Description: In User Settings Edit
FilePath: /BERT-pytorch-old/bert_pytorch/trainer/pretrain.py
'''

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from ..model import BERTLM, BERT
import tqdm

from .warmup import WarmUpLR


class BERTTrainer:
    """
    BERTTrainer make the pretrained BERT model with one LM training method.
    """
    def __init__(self, bert: BERT, vocab_size: int, train_dataloader: DataLoader=None, test_dataloader: DataLoader = None, lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, with_cuda: bool = True, batch_size=128, checkpoint_path = None, contrastive=True, alpha=1, gamma=1, warmup_epochs=5, gradient_accum=1, pair=False, lr_gamma=0.999, lr_patience=10, lr_adjust_gamma=0.9, sfile=None):
        '''
        @msg:  
        @params:  
        @return:  
        '''        
        # Setup cuda device for BERT training, argument -c, --cuda should be true
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")
        print("\t use cuda:",cuda_condition)
        self.contrastive = contrastive
        # This BERT model will be saved every epoch
        self.bert = bert

        # Initialize the BERT Language Model, with BERT model
        self.model = BERTLM(bert, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if torch.cuda.device_count() > 1:
            print("Using %d GPUS for BERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.batch_size = batch_size
        self.epoch = 0
        self.lr = lr
        self.weight_decay = weight_decay

        self.warmup_epochs = warmup_epochs
        self.gradient_accum = gradient_accum
        self.pair = pair
        self.lr_patience = lr_patience
        self.lr_gamma = lr_gamma
        self.lr_adjust_gamma = lr_adjust_gamma

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        self.warmup_scheduler = WarmUpLR(optimizer=self.optim, warmup_epochs=self.warmup_epochs, gamma=self.lr_gamma, patience=self.lr_patience, lr_adjust_gamma = self.lr_adjust_gamma, verbose=True)

        # Using Negative Log Likelihood Loss function for predicting the masked_token
        self.criterion = nn.NLLLoss(ignore_index=0)

        if self.contrastive:
            print('Joining contrastive learning with mask language model!')
            self.contrastive_loss = nn.CosineEmbeddingLoss(margin=0.0) #需要调优
            self.alpha = alpha
            self.gamma = gamma
        else:
            self.gamma = 1.0
            self.alpha = 1.0

        if checkpoint_path!= None:
            self.load_checkpoint(checkpoint_path)
        
        self.sf = sfile

        print("Total Parameters:\t\t", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(self.train_data,epoch)

    @torch.no_grad()
    def test(self,epoch):
        self.iteration(self.test_data,epoch, train=False)

    def cac_contrastive_loss(self, encode, label):
        idx = 0
        x1 = torch.empty((int(self.batch_size*(self.batch_size-1) / 2), self.bert.hidden)).cuda()
        x2 = torch.empty((int(self.batch_size*(self.batch_size-1) / 2), self.bert.hidden)).cuda()
        for i in range(self.batch_size):
            for j in range(i+1, self.batch_size):
                x1[idx] = encode[i]
                x2[idx] = encode[j]
                idx += 1
        return self.contrastive_loss(x1, x2, label)


    def iteration(self,data_loader,epoch,train=True):
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
        data_iter = tqdm.tqdm(enumerate(data_loader), desc="EP_%s:%d" % (str_code, epoch), total=len(data_loader), bar_format="{l_bar}{r_bar}")
        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for step, dataset in data_iter:
            # nbatch, [bert_inputs,bert_labels,segment_label,bert_is_next] = dataset
            [bert_inputs,bert_labels,segment_labels,is_next_labels,seq_names] = dataset

            is_same_origin = []
            bert_inputs = bert_inputs.cuda()
            bert_labels = bert_labels.cuda()
            segment_labels= segment_labels.cuda()
            is_next_labels = is_next_labels.cuda()
            
            loss = 0.0

            # # 1. forward the masked_lm model
            next_sent_output, mask_lm_output = self.model(bert_inputs,segment_labels)

            if self.pair == True:
                # 2-1. NLL(negative log likelihood) loss of is_next classification result
                next_loss = self.criterion(next_sent_output, is_next_labels)
                loss += next_loss
                correct = next_sent_output.argmax(dim=-1).eq(is_next_labels).sum().item()

                # next sentence prediction accuracy
                total_correct += correct
                total_element += is_next_labels.nelement()

            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), bert_labels)
            loss += mask_loss
            avg_loss += loss.item()
            # 3. backward and optimization only in train
            if train:
                loss /= self.gradient_accum
                loss.backward()
                if ((step + 1) % self.gradient_accum) == 0:
                    self.optim.step()
                    self.warmup_scheduler.step()
                    self.optim.zero_grad()

            # lr = self.warmup_scheduler.get_lr()[0]
            lr = self.optim.param_groups[0]['lr']
            if train:
                if self.sf!=None:
                    with open(self.sf,'a+') as f:
                        f.write("%s Epoch %d : Loss %2.6f : Avg Loss : %2.6f : lr : %2.9f\n" % (str_code,epoch,loss.item()*self.gradient_accum,avg_loss / (step+1), lr))

                if self.pair:
                    data_iter.set_description(f'{str_code.title()} Epoch {epoch} : Loss {loss.item() * self.gradient_accum: 2.6f} : Avg Loss : {avg_loss/(step+1): 2.6f} : total_acc : {total_correct * 100.0 / total_element :2.6f} : lr : {lr: 2.9f}')
                else:
                    data_iter.set_description(f'{str_code.title()} Epoch {epoch} : Loss {loss.item() * self.gradient_accum: 2.6f} : Avg Loss : {avg_loss/(step+1): 2.6f}  : lr : {lr: 2.9f}')
            else:
                if self.sf!=None:
                    with open(self.sf,'a+') as f:
                        f.write("%s Epoch %d : Loss %2.6f : Avg Loss : %2.6f : lr : %2.9f\n" % (str_code,epoch,loss.item(),avg_loss / (step+1), lr))

                if self.pair:
                    data_iter.set_description(f'{str_code.title()} Epoch {epoch} : Loss {loss.item() : 2.6f} : Avg Loss : {avg_loss/(step+1): 2.6f} : total_acc : {total_correct * 100.0 / total_element :2.6f} : lr : {lr: 2.9f}')
                else:
                    data_iter.set_description(f'{str_code.title()} Epoch {epoch} : Loss {loss.item() : 2.6f} : Avg Loss : {avg_loss/(step+1): 2.6f}  : lr : {lr: 2.9f}')

                    
    def save(self, epoch, path="output/bert_trained.model"):
        """
        Saving the current BERT model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path

        """
        state = {'model':self.model.state_dict(),'optimizer':self.optim.state_dict(),'epoch':epoch}
        torch.save(state, path)
        self.bert.to(self.device)
        # print(path)
        print("EP:%d Model Saved on:%s" % (epoch, path))
    
    def load_checkpoint(self,checkpoint_path):
        model_ckp= torch.load(checkpoint_path)
        self.model.load_state_dict (model_ckp['model'])
        self.epoch = model_ckp['epoch']
        print("Loading checkpoint!")
        self.optim.load_state_dict(model_ckp['optimizer'])
    