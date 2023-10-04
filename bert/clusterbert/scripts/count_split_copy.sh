#!/bin/bash
###
 # @Descripttion: 
 # @version: 
 # @Author: Yang jin
 # @Date: 2021-09-14 04:11:02
 # @LastEditors: Yang jin
 # @LastEditTime: 2021-10-26 03:57:52
### 

DATA_DIR_ROOT=/workspace/datasets/vamb-data

FILE_NAME=val.fa.500.frag.6mer.shf #7179447

# val-mag-gut.fa
SUB_DIR=/val

# FILE_NAME=all.lgs.f.train.fa.1000.fa.6mer
# 统计行数
A=`wc -l ${DATA_DIR_ROOT}/${FILE_NAME}` # 除以2才是序列的数量
A=${A% *} #获取第一个字段：行数 # #639035
B=2
val=`expr $A / $B`
echo "文件的总序列数目为：$val"

seq_len=512
BATCH_SIZE=8
FILE_SIZE=16

# 创建子目录
# rm ${DATA_DIR_ROOT}/bs_${BATCH_SIZE}sl_${seq_len}_pair/*
echo "将文件按batch_size切分为小文件，文件位置：${DATA_DIR_ROOT}${SUB_DIR}/bs_${BATCH_SIZE}sl_${seq_len}"
echo "运行结果文件生成：${DATA_DIR_ROOT}${SUB_DIR}/model_params_bs_${BATCH_SIZE}sl_${seq_len}"
mkdir ${DATA_DIR_ROOT}${SUB_DIR}
mkdir ${DATA_DIR_ROOT}${SUB_DIR}/bs_${BATCH_SIZE}sl_${seq_len}
mkdir ${DATA_DIR_ROOT}${SUB_DIR}/model_params_bs_${BATCH_SIZE}sl_${seq_len}

# 分割文件，每个子文件BATCH_SIZE行
split -d -a 6 -l ${FILE_SIZE} ${DATA_DIR_ROOT}/${FILE_NAME} ${DATA_DIR_ROOT}${SUB_DIR}/bs_${BATCH_SIZE}sl_${seq_len}/${FILE_NAME}.part
echo "切割完毕！"


