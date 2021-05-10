#!/bin/bash
work_dir=$(dirname $0)
cd $work_dir

DIR="./data/ImageNet/train"
ARCH="resnet50"
K=3000
WORKERS=8
WD=-4
LR=0.2
BATCH=1024
EXP="./experiments/UIC_R50"
EPOCH=500
SUFFIX='JPEG'

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS} --batch ${BATCH} \
  --epoch ${EPOCH} --suffix ${SUFFIX}
