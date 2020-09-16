#!/bin/bash
#
#$ -N ConvTasNet
#$ -S /bin/bash
#$ -o /pub/users/xpavlu10/log.txt
#$ -e /pub/users/xpavlu10/err.txt
#$ -q long.q@@gpu
#$ -l ram_free=3500M,mem_free=3500M,gpu=1

cd /pub/users/xpavlu10/conv-tasnet

unset PYTHONHOME
unset PYTHONPATH

source ../anaconda3/bin/activate
conda --version
echo "Enviroment activated"
python --version

set -eu
cpt_dir=exp/$(date +"%y-%m-%d-%H-%M-%S")

epochs=100
# constrainted by GPU number & memory
batch_size=32
cache_size=16
./nnet/train.py \
  --gpu 0 \
  --epochs $epochs \
  --batch-size $batch_size \
  --checkpoint $cpt_dir \
  > ./$cpt_dir.train.log 2>&1