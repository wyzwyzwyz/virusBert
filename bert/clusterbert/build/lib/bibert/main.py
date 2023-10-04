'''
Description: 
Version: 
Author: Yang jin
Date: 2021-10-13 07:18:53
LastEditors: Yang jin
LastEditTime: 2021-10-19 04:04:40
'''
import argparse
import os.path as osp

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from .model.bert import BERTEncoder

from .dataset import WordVocab,BERTDataset

from .model import config as cg
from .model import classifier as clf

from .model import BERT

from .trainer import BERTTrainer,TrainClassifier

def collate_fn_bert(batch):
    inputs, labels, tmp_segment_labels, tmp_is_next_labels, seq_names = [], [], [], [], []

    for [bert_input, bert_label, segment_label, is_next_label, seq_name] in batch:
        inputs.append(bert_input)
        labels.append(bert_label)
        tmp_segment_labels.append(segment_label)
        tmp_is_next_labels.append(is_next_label)
        for _ in seq_name:
            seq_names.append(_)

    bert_inputs = torch.cat(inputs, dim=0)
    bert_labels = torch.cat(labels, dim=0)
    segment_labels = torch.cat(tmp_segment_labels, dim=0)
    is_next_labels = torch.cat(tmp_is_next_labels, dim=0)

    return bert_inputs, bert_labels, segment_labels, is_next_labels, seq_names

def create_main_arg(parser):
    # 通用参数
    parser.add_argument("--mode", type=str, default='pretrain')

    parser.add_argument("-o", "--output_dir", type=str, default="./tmp")

    parser.add_argument("-w", "--num_workers", type=int, default=16)
    parser.add_argument("-lgg", "--logging_path", type=str,
                        default="./data/logging.log")

    parser.add_argument("--with_cuda", type=bool, default=True)

def create_bert_arg(parser):
    # BERT 的相关超参数
    '''
    data 相关参数
        vocab_path :词典的保存路径
    '''
    parser.add_argument("-c", "--train_dataset", type=str, default=None)
    parser.add_argument("-t", "--test_dataset", type=str, default=None)

    parser.add_argument("--corpus_lines_train", type=int)
    parser.add_argument("--corpus_lines_test", type=int)

    parser.add_argument("-v", "--vocab_path", type=str, default=None)

    # 500 is sequences' length ,but there is longer than sequences' length because of the special marker
    parser.add_argument("-s", "--seq_len", type=int, default=512)

    parser.add_argument("-b", "--batch_size", type=int, default=16)

    '''
    BERT 模型参数
        hidden:隐含层维度
        layers:transformer encode层层数
        attn_heads:attention 头数量
    '''
    parser.add_argument("-hs", "--hidden", type=int, default=768)
    parser.add_argument("-l", "--layers", type=int, default=6)
    parser.add_argument("-a", "--attn_heads", type=int, default=12)

    '''
    BERT pretrained 训练参数
    '''
    parser.add_argument("--pair", action="store_true",
                        help="input is couple or single")

    parser.add_argument("-p", "--checkpoint", type=str, default=None)

    parser.add_argument("-e", "--epochs", type=int, default=3)


    # ----mask 策略----
    parser.add_argument("-mn", "--mask_noise", type=float, default=0.1)
    parser.add_argument("-ml", "--mask_length", type=float, default=0.1)

    parser.add_argument("-lm", "--long_mask",
                        action="store_true", help="long masked or not")
    # 连续mask的word数量服从的正态分布
    parser.add_argument("-lmm", "--long_mask_mu", type=float,
                        default=6)  # if long_mask有效， 平均值mu
    parser.add_argument("-lms", "--long_mask_sgm", type=float,
                        default=3)  # if long_mask有效，方差sgm

    parser.add_argument("-dm", "--dynamic_mask",
                        action="store_true", help="dynamic masked or not")
    parser.add_argument("-mr", "--mask_rate", type=float,
                        default=0.1)  # 初始的mask 词的比列
    # if dynamic_mask有效，每个epoch后mask rate增长的量
    parser.add_argument("-incr", "--increment", type=float, default=0.1)
    # ----end----

    '''
    动态学习率 warmup 参数
        warnup_batchs:热身的batch 数目
        gradient_accum：累积梯度
        
    '''
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-8)

    parser.add_argument("-we", "--warmup_batchs", type=int, default=50)
    parser.add_argument("-ga", "--gradient_accum", type=int, default=128)

    parser.add_argument("-lrg", "--lr_gamma", type=float,
                        default=0.999)  # 指数衰减学习率下 衰减系数
    parser.add_argument("-adg", "--lr_adjust_gamma",
                        type=float, default=0.9)  # 学习率改变
    # 学习率在lr_patience 个loss 不降低的情况下，进行调整
    parser.add_argument("-lrp", "--lr_patience", type=int, default=10)

    '''
    Contrastive learning 参数
        loss = alpha*lcon + gamma*lmask
    '''
    parser.add_argument("--contrastive", action="store_true",
                        help="contrastive or not")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=1.0)

def create_classifier_arg(parser):
    parser.add_argument('--model_name', type=str, default="CNN")

    parser.add_argument("--finetuning",
                        action="store_true", help="finetune or not")

    parser.add_argument("--virus_flag",
                        action="store_true", help="input is virus genome or cds")

    # 将短fragment所获取的embed 拼接回长序列的策略
    parser.add_argument("--concat_strategy", type=str, default="MEAN")

    # 将BERTPooler 的策略，一般CLS，MAX
    parser.add_argument("--pooling_strategy", type=str, default="MAX")

    parser.add_argument("--input_dim",type=int,default = 768)

    parser.add_argument("--nlabels",type=int,default = 2)

    parser.add_argument("--cepochs", type=int, default=5)

    parser.add_argument("--dropout", type=float, default=0.2)

    parser.add_argument("--ccheckpoint", type=str,default=None)

    # In CNN,DNN both have layers_hidden param
    parser.add_argument("--layers_hidden", type=list, default=[1024,512,256])

    '''
    args for CNN(resnet) model config
    '''
    parser.add_argument("--in_channels",type=int,default = 768)

    parser.add_argument("--kernel_size",type=int,default = 3)
    parser.add_argument("--padding",type=int,default = 3)
    parser.add_argument("--stride",type=int,default = 2)



    '''
    args for LSTM model config
    '''

    '''
    adam优化器参数
    '''
    parser.add_argument('--clr', type=float, default=1e-3)
    parser.add_argument('--cadam_weight_decay', type=float, default=0)
    parser.add_argument('--cadam_beta1', type=float, default=0.9)
    parser.add_argument('--cadam_beta2', type=float, default=0.99)

    parser.add_argument("-cwe", "--cwarmup_batchs", type=int, default=50)
    # 由于样本量有限，训练分类任务中不需要累积梯度
    parser.add_argument("-cga", "--cgradient_accum", type=int, default=1)

    parser.add_argument("--cbatch_size", type=int, default=32)

    parser.add_argument("-clrg", "--clr_gamma", type=float,
                        default=0.99)  # 指数衰减学习率下 衰减系数
    parser.add_argument("-cadg", "--clr_adjust_gamma",
                        type=float, default=0.95)  # 学习率改变

    # 学习率在lr_patience 个loss 不降低的情况下，进行调整
    parser.add_argument("-clrp", "--clr_patience", type=int, default=10)


def dataset_builder(args, epoch, batch_size, mode):
    '''
    @msg:  
    @params:  
        epoch:dynamic mask中的参数需要根据epoch来确定
        batch_size:必须是8的倍数
        mode:pretrain模式下，采用mask技术 或者 classifer模式下，不采用mask，直接将序列输入进BERT
    @return: 
        dataset dataloader 
    '''    
    print("Loading Vocab:\t", args.vocab_path)
    vocab = WordVocab.load_vocab(args.vocab_path)
    print("Vocab Size:\t", len(vocab))

    if args.train_dataset is not None:
        print("Loading Train Dataset", args.train_dataset)
        train_dataset = BERTDataset(
            corpus_path_base=args.train_dataset, vocab=vocab, seq_len=args.seq_len, batch_size=batch_size, corpus_lines=args.corpus_lines_train, \
                long_mask=args.long_mask, long_mask_mu=args.long_mask_mu, long_mask_sgm=args.long_mask_sgm,\
                 mask_length=args.mask_length+(args.increment*epoch)/args.epochs, mask_noise=args.mask_noise, mask_rate=args.mask_rate, pair=args.pair, mode=mode
        )
        print("\tThe number of Batches:\t", len(train_dataset))
        print("\tBatch size:\t",args.batch_size)
        print("\tSeq len:\t", args.seq_len)
    else:
        train_dataset = None

    if args.test_dataset is not None:
        print("Loading Test Dataset", args.test_dataset)
        test_dataset = BERTDataset(
            corpus_path_base=args.test_dataset, vocab=vocab, seq_len=args.seq_len, batch_size=batch_size, corpus_lines=args.corpus_lines_test,  \
                long_mask=args.long_mask, long_mask_mu=args.long_mask_mu, long_mask_sgm=args.long_mask_sgm,\
                 mask_length=args.mask_length, mask_noise=args.mask_noise, mask_rate=args.mask_rate,  pair=args.pair, mode=mode
        )
        print("\tThe number of Batches:\t", len(test_dataset))
    else:
        test_dataset = None

    print("Creating Dataloader")

    assert batch_size%8 == 0

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=int(batch_size/8),
        num_workers=args.num_workers,
        collate_fn=collate_fn_bert,
        pin_memory=True,
        shuffle=True
    ) if train_dataset is not None else None

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=int(batch_size/8),
        num_workers=args.num_workers, collate_fn=collate_fn_bert,
        pin_memory=True,
        shuffle=True
    ) if test_dataset is not None else None

    return vocab, train_data_loader, test_data_loader

def trainer_builder(args, output_path, epoch, batch_size, mode="pretrain"):
    print('Output Path:', output_path)
    vocab, train_data_loader, test_data_loader = dataset_builder(
        args, epoch, batch_size, mode)
    print("Building BERT model")

    config = cg.BERTConfig(
        vocab_size=len(vocab),
        hidden=args.hidden,
        n_layers=args.layers,
        attn_heads=args.attn_heads
    )
    bert = BERT(config)
    print('\tnHidden:\t\t', bert.hidden)
    print('\tnLayers:\t\t', bert.n_layers)
    print('\tnAttn heads:\t\t', bert.attn_heads)

    print("Creating BERT Trainer!")
    trainer = BERTTrainer(
        bert,
        len(vocab),
        train_dataloader=train_data_loader, test_dataloader=test_data_loader,
        lr=args.lr, betas=(
            args.adam_beta1, args.adam_beta2
        ),
        weight_decay=args.adam_weight_decay,
        with_cuda=args.with_cuda,
        batch_size=batch_size,
        checkpoint_path=args.checkpoint,
        contrastive=args.contrastive,
        alpha=args.alpha,
        gamma=args.gamma,
        sfile=args.logging_path,
        gradient_accum=args.gradient_accum,
        pair=args.pair,
        warmup_epochs=args.warmup_batchs,
        lr_gamma=args.lr_gamma,
        lr_patience=args.lr_patience,
        lr_adjust_gamma=args.lr_adjust_gamma
    )

    if args.checkpoint:
        # 第二个epoch 就是从这里加载上一次的模型结果
        print("Loading BERT from checkpoint:", args.checkpoint)
    else:
        print("Training BERT from scratch!")
        print('\tnHidden:\t{}'.format(bert.hidden))
        print('\tnLayers:\t{}'.format(bert.n_layers))
        print('\tnAttn heads:\t{}'.format(bert.attn_heads))

        print("\tLearning rate:\t", trainer.lr)
        print('\tAdam weight decay:\t{}'.format(trainer.weight_decay))
        print('\tWarmup_epochs:\t{}'.format(trainer.warmup_epochs))
        print('\tGradient_accum:\t{}'.format(trainer.gradient_accum))
  
    return trainer, train_data_loader, test_data_loader, config

def run_bert(args, output_path):
    # 构建第一个epoch 的训练器
    trainer, train_data_loader, test_data_loader, config = trainer_builder(
        args, output_path, 0, args.batch_size, "pretrain")

    for epoch in range(args.epochs):
        if train_data_loader is not None:
            if args.dynamic_mask and epoch > 0:
                args.checkpoint = output_path + '.ep%d.pt' % (epoch - 1)
                trainer, train_data_loader, test_data_loader, config = trainer_builder(
                    args, output_path, epoch, args.batch_size, "pretrain")
            trainer.train(epoch)
            model_path = output_path + '.ep%d.pt' % epoch
            trainer.save(epoch, model_path)

        if test_data_loader is not None:
            trainer.test(epoch)


def run_classifer(args, output_path):
    '''
    @parma:
        mode:对于bert而言，无需mask序列，加载数据时mode为test
    '''
    trainer, train_dl, test_dl, config = trainer_builder(
        args, output_path, 0, args.batch_size, mode="test")
    
    print('Creating BERT Classifier!')
    # biClassifier = classifier_builder(**args.__dict__)

    bert_encoder = BERTEncoder(trainer.bert,config)

    train_clf = TrainClassifier(bert_encoder,train_dl,test_dl,**args.__dict__)

    print('Output path: ', output_path)
    if args.ccheckpoint != None:
        print('Loading Classifer from checkpoint: ', args.ccheckpoint)
    else:
        print('Training Classifer from scratch!')
        print('\tModel name:\t',args.model_name)
        print('\tLayers_hidden\t',args.layers_hidden)
        print('\tBERT Pooler strategy:\t',train_clf.pooling_strategy)
        print('\tBatch size:\t',train_clf.batch_size)
        print('\tNlabels:\t', train_clf.nlabels)
        print('\tLearning rate:\t', train_clf.lr)
        print('\tAdam weight decay:\t', train_clf.weight_decay)
        print('\tWarmup_epochs:\t',train_clf.warmup_epochs)
        print('\tGradient_accum:\t',train_clf.gradient_accum)
        
    if args.logging_path:
        # 参数写入logging file
        with open(args.logging_path, 'a+') as filehandle:
            filehandle.write('BERT Output Path:{}\n'.format(output_path))
            filehandle.write('\tModel name:\t{}\n'.format(args.model_name))
            filehandle.write(
                '\tBERT Pooler strategy:\t{}\n'.format(train_clf.pooling_strategy))
                
            filehandle.write('\tBatch size:\t{}\n'.format(train_clf.batch_size))
            filehandle.write('\tNlabels:\t{}\n'.format(train_clf.nlabels))
            filehandle.write('\tLearning rate:\t{}\n'.format(train_clf.lr))
            filehandle.write('\tAdam weight decay:\t{}\n'.format(train_clf.weight_decay))
            filehandle.write('\tWarmup_epochs:\t{}\n'.format(train_clf.warmup_epochs))
            filehandle.write('\tGradient_accum:\t{}\n'.format(train_clf.gradient_accum))

    for epoch in range(args.cepochs):
        if train_dl is not None:
            train_clf.train(epoch)
            model_path = output_path + '.ep%d.pt' % epoch
            train_clf.save(epoch, model_path)

        if test_dl is not None:
            train_clf.test(epoch)


def train(args):
    if args.mode == "pretrain":
        output_path = osp.join(
            args.output_dir, f'pre_h{args.hidden}-l{args.layers}-a{args.attn_heads}-b{args.batch_size}-lr{args.lr}-lm{args.long_mask}-pair{args.pair}-ga{args.gradient_accum}')
        run_bert(args, output_path)

    elif args.mode == "classifier":
        output_path = osp.join(
            args.output_dir, f'cl-h{args.hidden}-l{args.layers}-a{args.attn_heads}-cb{args.cbatch_size}-lr{args.lr}-clr{args.clr}'
        )
        run_classifer(args, output_path)

    else:
        print("Mode error!")

def main():
    parser = argparse.ArgumentParser()
    create_main_arg(parser)
    create_bert_arg(parser)
    create_classifier_arg(parser)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
