#!/bin/bash
work_dir=$(dirname $0)
cd $work_dir

DATA="./data/ImageNet"
MODEL="./experiments/UIC_R50/checkpoint_latest.pth.tar"
EXP="./experiments/UIC_R50/linear_eval"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0 python eval_linear.py --model ${MODEL} --data ${DATA} --lr 0.1 \
  --wd 0 --verbose --exp ${EXP} --workers 8 --tencrops