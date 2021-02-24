import argparse

def run(args):
    with open("./expscripts/{}.sh".format(args.name), "w") as file:
        file.write(
'#!/bin/bash \n\
# \n\
#$ -N {} \n\
#$ -S /bin/bash \n\
#$ -o /pub/users/xpavlu10/log.txt \n\
#$ -e /pub/users/xpavlu10/err.txt \n\
#$ -q long.q@@gpu \n\
#$ -l ram_free=3500M,mem_free=3500M,gpu=1 \n\
\n\
ulimit -a\n\
\n\
ulimit -t unlimited  \n\
\n\
ulimit -a \n\
\n\
cd /pub/users/xpavlu10/conv-tasnet \n\
\n\
unset PYTHONHOME \n\
unset PYTHONPATH \n\
\n\
source ../anaconda3/bin/activate \n\
conda --version \n\
echo "Enviroment activated" \n\
python --version \n\
\n\
set -eu \n\
commment="{}" \n\
cpt_dir=exp/$(date +"%y-%m-%d-%H-%M-%S")_$comment \n\
\n\
gpu=0 \n\
mixofmix=1 \n\
known_percent={} \n\
epochs={} \n\
# constrainted by GPU number & memory \n\
batch_size={} \n\
cache_size=8 \n\
./nnet/train.py \\ \n\
--gpu $gpu \\ \n\
--epochs $epochs \\ \n\
--batch-size $batch_size \\ \n\
--checkpoint $cpt_dir \\ \n\
--mixofmix $mixofmix \\ \n\
--known_percent $known_percent \\ \n\
> ./$cpt_dir.train.log 2>&1'.format(args.name, args.comment, args.known_percent, args.epochs, args.batch_size))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Generates experiment script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--name",
                        type=str,
                        default="",
                        help="Name of the experiment script")
    parser.add_argument("--comment",
                        type=str,
                        default="",
                        help="Comment for current experiment")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--epochs",
                        type=int,
                        default=150,
                        help="Number of training epochs")
    parser.add_argument("--known-percent",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    args = parser.parse_args()
    
    run(args)