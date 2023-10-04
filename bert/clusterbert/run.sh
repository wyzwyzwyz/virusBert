#!/bin/bash
###
 # @Description: 
 # @Version: 
 # @Author: Yang jin
 # @Date: 2021-10-18 08:28:06
 # @LastEditors: Yang jin
 # @LastEditTime: 2022-06-24 08:01:48
### 

#---- split seqs into 500 fragments,then split each fragment into overlapped 6mer----
./fasta2kmers /workspace/datasets/cds/Euk/tra.fna.shf /workspace/datasets/cds/Euk/tra.fna.shf.500.frag 0 500
./fasta2kmers /workspace/datasets/cds/Euk/tra.fna.shf.500.frag  /workspace/datasets/cds/Euk/tra.fna.shf.500.frag.6mer 1 6

#----build vocab, -m ：word出现的最小频数----
# logging
# Building Vocab
# 685726it [01:40, 6827.26it/s]
# VOCAB SIZE: 4102
awk 'NR%2==0' tra.fna.shf.500.frag.6mer > tra.fna.shf.500.frag.6mer.corpus
bibert-vocab -c /workspace/datasets/cds/Euk/tra.fna.shf.500.frag.6mer.corpus -o /workspace/MG-tools/BiBERT/data/cds.vocab -m 100
#---split big file into small file by batchsize----
#logging
# 文件的总序列数目为：685726
# 将文件按batch_size切分为小文件，文件位置：/workspace/datasets/cds/Euk//1015/bs_8sl_512
# 运行结果文件生成：/workspace/datasets/cds/Euk//1015/model_params_bs_8sl_512


# #----BERT pretrain----
# # bibert -o  /workspace/datasets/cds/Euk/1015/model_params_bs_8sl_512/ -v /workspace/MG-tools/Dbibert/BiBERT/data/cds.vocab -c /workspace/datasets/cds/Euk/1015/bs_8sl_512/tra.fna.shf.500.frag.6mer --corpus_lines_train 685726 -t /workspace/datasets/cds/Euk/val_1015/bs_8sl_512/val.fna.shf.500.frag.6mer --corpus_lines_test 211542 -b 32 -ga 64  -lgg /workspace/datasets/cds/Euk/1015/model_params_bs_8sl_512/bert1016.log --long_mask  -dm  --lr 5e-4  -we 15  -e 10
# bibert -o  /workspace/MG-DL/datasets/toy-low/1110/model_params_bs_8sl_512 -v /workspace/MG-tools/Dbibert/BiBERT/data/cds.vocab -c /workspace/datasets/cds/Euk/1015/bs_8sl_512/tra.fna.shf.500.frag.6mer --corpus_lines_train 685726 -t /workspace/datasets/cds/Euk/val_1015/bs_8sl_512/val.fna.shf.500.frag.6mer --corpus_lines_test 211542 -b 32 -ga 64  -lgg /workspace/datasets/cds/Euk/1015/model_params_bs_8sl_512/bert1016.log --long_mask  -dm  --lr 5e-4  -we 15  -e 10

#预训练3mer
bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus3/bs_8sl_512/corpus.fa.500.frag.4mer -o /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/0330_768.log -v /workspace/vamb-data/1229fa/pool4mer.vocab -hs 768 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0
#预训练4mer
bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus4/bs_8sl_512/corpus.fa.500.frag.4mer -o /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/0320_256.log -v /workspace/vamb-data/1229fa/pool4mer.vocab -hs 256 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0
#预训练5mer
 bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus5/bs_8sl_512/corpus.fa.500.frag.5mer -o /workspace/vamb-data/1229fa/corpus5/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus5/model_params_bs_8sl_512/0330_768.log -v /workspace/vamb-data/1229fa/pool5mer.vocab -hs 768 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0
#预训练7mer
bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus7/bs_8sl_512/corpus.fa.500.frag.7mer -o /workspace/vamb-data/1229fa/corpus7/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus7/model_params_bs_8sl_512/0330_768.log -v /workspace/vamb-data/1229fa/pool7mer.vocab -hs 768 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0


# #----Cluster----####docker-8026服务器8026端口映射目录/data1/bert-data clbert命令####
# bibert --mode cluster -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/vamb-data/skin/1110/model_params_bs_8sl_512  --embed_path /workspace/vamb-data/skin/encode --corpus_lines_test 4031573 -t /workspace/vamb-data/skin/1110/bs_8sl_512/cutted-contigs.fa.500.100str.6mer  -v /workspace/1026.vocab 

bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/vamb-data/skin/1202/model_params_bs_256sl_512  --embed_path /workspace/vamb-data/skin/1222/encode-all-cls --corpus_lines_test 3694843 -t /workspace/vamb-data/skin/1202/bs_256sl_512/contigs.fa.500.frag.6mer  -v /workspace/1026.vocab 

# conda activate clbert

bibert --mode cluster --output_all_encoded -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/vamb-data/gi/1202/model_params_bs_256sl_512  --embed_path /workspace/vamb-data/gi/1222/encode-all-cls --corpus_lines_test 3484604 -t /workspace/vamb-data/gi/1202/bs_256sl_512/contigs.fa.500.frag.6mer  -v /workspace/1026.vocab 

conda activate clbert

######## toy-high
bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 32 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/CAMI-contigs/toy-high/sample/s4/1202/model_params_bs_32sl_512  --embed_path  /workspace/CAMI-contigs/toy-high/sample/s4/1202/encode-02-cls/ --corpus_lines_test 3441251  -t /workspace/CAMI-contigs/toy-high/sample/s4/1202/bs_32sl_512/contigs.fa.500.frag.6mer  -v /workspace/1026.vocab 

bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 32 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/CAMI-contigs/toy-high/sample/s5/1202/model_params_bs_32sl_512  --embed_path  /workspace/CAMI-contigs/toy-high/sample/s5/1202/encode-02-cls/ --corpus_lines_test 3439212  -t /workspace/CAMI-contigs/toy-high/sample/s5/1202/bs_32sl_512/contigs.fa.500.frag.6mer  -v /workspace/1026.vocab 

###
bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/mother/megahit/3-1202/model_params_bs_256sl_512  --embed_path  /workspace/mother/megahit/3-1202/encode-02-cls/ --corpus_lines_test 8817671 -t /workspace/mother/megahit/3-1202/bs_256sl_512/3-contigs.fa.500.frag.6mer  -v /workspace/1026.vocab

### 
bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/pre_h512-l12-a8-b32-lr0.003-lmTrue-pairFalse-ga20-2021_10_29_23_46_45.ep19.pt -o /workspace/mother/megahit/4-1202/model_params_bs_128sl_512  --embed_path  /workspace/mother/megahit/4-1202/encode-02-cls/ --corpus_lines_test 6966001 -t /workspace/mother/megahit/4-1202/bs_128sl_512/4-contigs.fa.500.frag.6mer  -v /workspace/1026.vocab

###
bibert --mode cluster  --output_all_encoded -l 12 -a 8 --hidden 512 -b 256 -p /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/newest.ep0step22000.pt  -o /workspace/vamb-data/airways/1202/corpus4/encode-02  --embed_path  /workspace/vamb-data/airways/1202/corpus4/encode-02-cls/ --corpus_lines_test 3406788 -t /workspace/vamb-data/airways/1202/corpus4/bs_256sl_512/contigs.fa.500.frag.4mer  -v /workspace/vamb-data/1229fa/pool4mer.vocab


###
bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus7/bs_8sl_512/corpus.fa.500.frag.7mer -o /workspace/vamb-data/1229fa/corpus7/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus7/model_params_bs_8sl_512/0330_768.log -v /workspace/vamb-data/1229fa/pool4mer.vocab -hs 768 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0 


bibert --mode pretrain -c /workspace/vamb-data/1229fa/corpus4/bs_8sl_512/corpus.fa.500.frag.4mer -o /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/ --logging_path /workspace/vamb-data/1229fa/corpus4/model_params_bs_8sl_512/0330_768.log -v /workspace/vamb-data/1229fa/pool4mer.vocab -hs 768 -l 12 -a 8 -s 512 -b 32 -e 6 -w 32 -ga 20 --pooling_strategy MAX --with_cuda True --corpus_lines_train 11687989 --corpus_lines_test 0  -we 40 --lr 3e-3 --adam_weight_decay 1e-3 --adam_beta1 0.9 --adam_beta2 0.999 -lm -dm -lmm 6 -lms 3 -incr 0.1 --alpha 1.0 --gamma 1.0 -lrg 0.998 -adg 0.95 -lrp 10 -mn 0.0 