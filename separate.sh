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
./nnet/separate.py ./exp/20-10-27-10-58-19 --input ../min_dataset/tt/mix.scp --gpu 1 --dump-dir ./out/$(date +"%y-%m-%d-%H-%M-%S") --plot 1