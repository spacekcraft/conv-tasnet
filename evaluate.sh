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

for arg in $@; do
    ./nnet/compute_si_snr.py ./out/$arg ../min_dataset/tt/s1.scp,../min_dataset/tt/s2.scp --mixofmix 1
done

