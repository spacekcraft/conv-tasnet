#!/bin/bash
#
#$ -N k100ConvTasNet
#$ -S /bin/bash
#$ -o /pub/users/xpavlu10/log.txt
#$ -e /pub/users/xpavlu10/err.txt
#$ -q long.q@@gpu
#$ -l ram_free=3500M,mem_free=3500M,gpu=1

ulimit -a

ulimit -t unlimited  

ulimit -a

cd /pub/users/xpavlu10/conv-tasnet

unset PYTHONHOME
unset PYTHONPATH

source ../anaconda3/bin/activate
conda --version
echo "Enviroment activated"
python --version

set -eu
cpt_dir=exp/$(date +"%y-%m-%d-%H-%M-%S")

gpu=0
mixofmix=1
known_percent=100
epochs=100
# constrainted by GPU number & memory
batch_size=16
cache_size=8
./nnet/train.py \
  --gpu $gpu \
  --epochs $epochs \
  --batch-size $batch_size \
  --checkpoint $cpt_dir \
  --mixofmix $mixofmix \
  --known_percent $known_percent\
  > ./$cpt_dir.train.log 2>&1