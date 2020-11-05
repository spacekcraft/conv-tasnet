#!/bin/bash

ulimit -a

ulimit -t unlimited  

ulimit -a

cd /pub/users/xpavlu10/conv-tasnet

unset PYTHONHOME
unset PYTHONPATH

source ../anaconda3/bin/activate
conda --version
echo "Enviroment activated"

outDir=./out/$(date +"%y-%m-%d-%H-%M-%S")
./nnet/separate.py ./exp/$1 --input ../min_dataset/tt/mix.scp --gpu 1 --dump-dir $outDir --plot 1
./nnet/compute_si_snr.py $outDir ../min_dataset/tt/s1.scp,../min_dataset/tt/s2.scp --mixofmix 1