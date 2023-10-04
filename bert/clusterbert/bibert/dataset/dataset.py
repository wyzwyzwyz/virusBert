'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-13 08:58:04
LastEditors: CAI Zhijie
LastEditTime: 2022-02-14 07:56:27
'''
from torch.utils.data import Dataset
import torch
import random
import math
import os
import numpy  as np


class BERTDataset(Dataset):
    def __init__(self, corpus_path_base, vocab, seq_len, batch_size, corpus_lines, \
        long_mask, long_mask_mu, long_mask_sgm, mask_length=0.1, mask_noise=0.05, mask_rate=0.9, \
        pair=False, mode="pretrain"):
        
        self.corpus_path_base = corpus_path_base
        self.corpus_lines = corpus_lines
        self.seq_len = seq_len

        self.vocab = vocab

        self.long_mask = long_mask
        self.long_mask_mu = long_mask_mu
        self.long_mask_sgm = long_mask_sgm

        self.mask_length = mask_length
        self.mask_noise = mask_noise
        self.mask_rate = mask_rate

        self.batch_size = batch_size
        self.mode = mode

        self.pair = pair # if True:couple sentence 模式

        self.datas = []
        self.seq_names = []

    def load_small_corpus(self, corpus_path, encoding="utf-8"):
        """Load large corpus file in batches.
        params:
            item: which batches,random.
            corpus_path: No item-th batches 's path on the memory.
        """
        self.datas.clear()
        self.seq_names.clear()
        with open(corpus_path, "r", encoding=encoding) as f:
            if self.pair:
                for line in f:
                    if line.startswith('>'):
                        self.seq_names.append(line.strip('>').strip())
                    else:
                        self.datas.append(line.strip().split('\t'))
            else:
                for line in f:
                    if line.startswith(">"):
                        self.seq_names.append(line.strip('>').strip())
                    else:
                        self.datas.append(line.strip())

    def __len__(self):
        # return len(self.datas)
        print("长度为：")
        print(math.ceil((self.corpus_lines/self.batch_size)))
        return math.ceil((self.corpus_lines/self.batch_size))
        
        # return math.ceil((self.corpus_lines/8))

    def random_sent(self, index):
        t1, t2 = self.datas[index][0], self.datas[index][1]

        # output_text, label(isNotNext:0, isNext:1)
        if random.random() > 0.5:
            return t1, t2, 1
        else:
            return t1, self.get_random_line(index), 0

    def next_sent(self, index):
        t1, t2 = self.datas[index][0], self.datas[index][1]
        return t1, t2, 1

    def get_random_line(self, index):
        max_lines = len(self.datas)

        rdm = random.randint(0, max_lines-1)
        while rdm == index:
            rdm = random.randint(0, max_lines-1)

        return self.datas[rdm][1]

    def real_word(self, sentence):
        tokens = sentence.split()
        output_label = []
        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)
            output_label.append(self.vocab.stoi.get(
                token, self.vocab.unk_index))

        return tokens, output_label

    def random_word(self, sentence):
        tokens = sentence.split()
        input_ids, output_label = [], []
        vl = len(tokens)
        if not self.long_mask:
            for i, token in enumerate(tokens):
                prob = random.random()
                if prob < self.mask_length:
                    prob /= self.mask_length
                    # 90% randomly change token to mask token
                    if prob < self.mask_rate:
                        input_ids.append(self.vocab.mask_index)

                    # 5% randomly change token to random token
                    elif self.mask_rate <= prob < self.mask_rate + self.mask_noise:
                        input_ids.append(random.randrange(len(self.vocab)))

                    # 5% randomly change token to current token
                    elif prob >= (self.mask_rate + self.mask_noise):
                        input_ids.append(self.vocab.stoi.get(
                            token, self.vocab.unk_index))

                    output_label.append(self.vocab.stoi.get(
                        token, self.vocab.unk_index))

                else:
                    input_ids.append(self.vocab.stoi.get(
                        token, self.vocab.unk_index))

                    output_label.append(0) # output 不用预测

        else:
            p, q = 0, 0
            for i, token in enumerate(tokens):
                if i < q:
                    continue
                prob = random.random()
                if prob < self.mask_length / self.long_mask_mu:
                    prob /= (self.mask_length / self.long_mask_mu)

                    window = np.int(random.gauss(
                        self.long_mask_mu, self.long_mask_sgm))
                    p, q = i, min(vl, i+window+1)
                    # 90% randomly change token to mask token
                    if prob < self.mask_rate:
                        input_ids.extend(
                            [self.vocab.mask_index for _ in range(p, q)])

                    # 5% randomly change token to random token
                    elif self.mask_rate <= prob < self.mask_rate + self.mask_noise:
                        input_ids.extend(
                            [random.randrange(len(self.vocab)) for _ in range(p, q)])

                    # 5% randomly change token to current token
                    elif prob >= self.mask_rate + self.mask_noise:
                        input_ids.extend([self.vocab.stoi.get(
                            tokens[_], self.vocab.unk_index) for _ in range(p, q)])

                    output_label.extend([self.vocab.stoi.get(
                        tokens[_], self.vocab.unk_index) for _ in range(p, q)])

                else:
                    input_ids.append(self.vocab.stoi.get(
                        token, self.vocab.unk_index))
                    output_label.append(0)

        return input_ids, output_label

    def __getitem__(self, item):
        '''
        Inputs are single sentences rather than couple sentence.
        '''
        self.datas.clear()
        self.seq_names.clear()
        if(os.path.exists(self.corpus_path_base + f".{item:08}")):
            self.load_small_corpus(self.corpus_path_base + f".{item:08}")
        else:
            self.load_small_corpus(self.corpus_path_base + f".000001")
      

        #self.load_small_corpus(self.corpus_path_base+f".part{item:08}")
        bert_inputs, bert_labels, segment_labels, is_next_labels = [], [], [], []

        assert len(self.seq_names) == len(self.datas)
        
        # if len(self.datas)!= self.batch_size:
        #     print("\n----There are no {}th file shorter than batch size {}----".format(item,self.batch_size))
        #     print("----The datas and seq names ' length is {}----".format(len(self.seq_names)))

        if self.pair:
            #在这里 基本不用couple模式
            for i in range(len(self.datas)):
                assert len(self.datas[i]) == 2
                if self.mode == "pretrain":

                    t1, t2, is_next_label = self.random_sent(i)
                    t1_random, t1_label = self.random_word(t1)
                    t2_random, t2_label = self.random_word(t2)

                    t1 = [self.vocab.sos_index] + \
                        t1_random + [self.vocab.eos_index]
                    t2 = t2_random + [self.vocab.eos_index]

                    t1_label = [self.vocab.pad_index] + \
                        t1_label + [self.vocab.pad_index]
                    t2_label = t2_label + [self.vocab.pad_index]

                    bert_input = (t1 + t2)[:self.seq_len]
                    bert_label = (t1_label + t2_label)[:self.seq_len]

                    segment_label = (
                        [1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

                    padding = [self.vocab.pad_index for _ in range(
                        self.seq_len - len(bert_input))]

                    bert_input.extend(padding), bert_label.extend(
                        padding), segment_label.extend(padding)

                    bert_inputs.append(bert_input)
                    bert_labels.append(bert_label)
                    is_next_labels.append(is_next_label)
                    segment_labels.append(segment_label)

                else:
                    t1, t2, is_next_label = self.next_sent(i)
                    t1_random, t1_label = self.real_word(t1)
                    t2_random, t2_label = self.real_word(t2)
                    # segment label is all zero for testing

                    t1 = [self.vocab.sos_index] + \
                        t1_random + [self.vocab.eos_index]
                    t2 = t2_random + [self.vocab.eos_index]

                    t1_label = [self.vocab.pad_index] + \
                        t1_label + [self.vocab.pad_index]
                    t2_label = t2_label + [self.vocab.pad_index]

                    bert_input = (t1 + t2)[:self.seq_len]
                    bert_label = (t1_label + t2_label)[:self.seq_len]

                    segment_label = ([0 for _ in range(len(bert_input))])
                    # segment_label = (
                    #     [1 for _ in range(len(t1))] + [2 for _ in range(len(t2))])[:self.seq_len]

                    padding = [self.vocab.pad_index for _ in range(
                        self.seq_len - len(bert_input))]

                    bert_input.extend(padding), bert_label.extend(
                        padding), segment_label.extend(padding)
                    # [CLS] tag = SOS tag, [SEP] tag = EOS tag
                    bert_inputs.append(bert_input)
                    bert_labels.append(bert_label)
                    is_next_labels.append(is_next_label)
                    segment_labels.append(segment_label)

        else:
            for i in range(len(self.datas)):
                sen = self.datas[i]

                if self.mode == "pretrain":
                    sen_random, sen_label = self.random_word(sen)

                    sen = [self.vocab.sos_index] + \
                        sen_random + [self.vocab.eos_index]

                    sen_label = [self.vocab.pad_index] + \
                        sen_label + [self.vocab.pad_index]

                    bert_input = (sen)[:self.seq_len]
                    bert_label = (sen_label)[:self.seq_len]
                    segment_label = ([0 for _ in range(len(sen))])[
                        :self.seq_len]
                    padding = [self.vocab.pad_index for _ in range(
                        self.seq_len - len(bert_input))]

                    bert_input.extend(padding), bert_label.extend(
                        padding), segment_label.extend(padding)

                    bert_inputs.append(bert_input)
                    bert_labels.append(bert_label)
                    segment_labels.append(segment_label)

                else:
                    sen_random, sen_label = self.real_word(sen)

                    # segment label is all zero for testing

                    sen = [self.vocab.sos_index] + \
                        sen_random + [self.vocab.eos_index]

                    sen_label = [self.vocab.pad_index] + \
                        sen_label + [self.vocab.pad_index]

                    segment_label = ([0 for _ in range(len(sen))])[
                        :self.seq_len]

                    bert_input = (sen)[:self.seq_len]
                    bert_label = (sen_label)[:self.seq_len]
                    # [CLS] tag = SOS tag, [SEP] tag = EOS tag
                    padding = [self.vocab.pad_index for _ in range(
                        self.seq_len - len(bert_input))]

                    bert_input.extend(padding), bert_label.extend(
                        padding), segment_label.extend(padding)

                    bert_inputs.append(bert_input)
                    bert_labels.append(bert_label)
                    is_next_labels.append(1)
                    segment_labels.append(segment_label)

        return torch.tensor(bert_inputs), torch.tensor(bert_labels), torch.tensor(segment_labels),  self.seq_names
