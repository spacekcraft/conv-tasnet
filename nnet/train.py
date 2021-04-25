#!/usr/bin/env python

# wujian@2018

import os
import pprint
import argparse
import random

from libs.trainer import SiSnrTrainer
from libs.trainer import MixtureOfMixturesTrainer
from libs.dataset import make_dataloader
from libs.utils import dump_json, get_logger

from conv_tas_net import ConvTasNet
from conf import trainer_conf, nnet_conf, train_data, dev_data, chunk_size
import pdb

logger = get_logger(__name__)


def run(args):
    train_data["knownPercent"] = args.known_percent
    dev_data["knownPercent"] = args.known_percent
    train_data["only_supervised"] = args.only_supervised
    dev_data["only_supervised"] = args.only_supervised
    gpuids = tuple(map(int, args.gpus.split(",")))

    nnet = ConvTasNet(**nnet_conf)
    if args.mixofmix == 0:
        logger.info("SisSnrTrainer")
        trainer = SiSnrTrainer(nnet,
                            gpuid=gpuids,
                            checkpoint=args.checkpoint,
                            resume=args.resume,
                            comment = args.comment,
                            log_dir = args.log_dir,
                            **trainer_conf)
    else:
        logger.info("MixtureOfMixturesTrainer")
        trainer = MixtureOfMixturesTrainer(nnet,
                            gpuid=gpuids,
                            checkpoint=args.checkpoint,
                            resume=args.resume,
                            comment = args.comment,
                            log_dir = args.log_dir,
                            **trainer_conf)
    logger.info("Known pecents "+str(dev_data["knownPercent"]))
    data_conf = {
        "train": train_data,
        "dev": dev_data,
        "chunk_size": chunk_size
    }
    for conf, fname in zip([nnet_conf, trainer_conf, data_conf],
                           ["mdl.json", "trainer.json", "data.json"]):
        dump_json(conf, args.checkpoint, fname)

    if args.mixofmix == 0:
        train_loader = make_dataloader(train=True,
                                    data_kwargs=train_data,
                                    batch_size=args.batch_size,
                                    chunk_size=chunk_size,
                                    num_workers=args.num_workers)
        dev_loader = make_dataloader(train=False,
                                    data_kwargs=dev_data,
                                    batch_size=args.batch_size,
                                    chunk_size=chunk_size,
                                    num_workers=args.num_workers)
    else:
        train_loader = make_dataloader(train=True,
                                    data_kwargs=train_data,
                                    batch_size=args.batch_size,
                                    chunk_size=chunk_size,
                                    num_workers=args.num_workers,
                                    mixofmix = True)
        dev_loader = make_dataloader(train=False,
                                    data_kwargs=dev_data,
                                    batch_size=args.batch_size,
                                    chunk_size=chunk_size,
                                    num_workers=args.num_workers,
                                    mixofmix = True)
    trainer.run(train_loader, dev_loader, num_epochs=args.epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to start ConvTasNet training, configured from conf.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpus",
                        type=str,
                        default="0,1",
                        help="Training on which GPUs "
                        "(one or more, egs: 0, \"0,1\")")
    parser.add_argument("--epochs",
                        type=int,
                        default=50,
                        help="Number of training epochs")
    parser.add_argument("--checkpoint",
                        type=str,
                        required=True,
                        help="Directory to dump models")
    parser.add_argument("--resume",
                        type=str,
                        default="",
                        help="Exist model to resume training from")
    parser.add_argument("--batch-size",
                        type=int,
                        default=16,
                        help="Number of utterances in each batch")
    parser.add_argument("--num-workers",
                        type=int,
                        default=4,
                        help="Number of workers used in data loader")
    parser.add_argument("--mixofmix",
                        type=int,
                        default=0,
                        help="Number of workers used in data loader")
    
    parser.add_argument("--only_supervised",
                        type=bool,
                        default=False,
                        help="Number of workers used in data loader")

    parser.add_argument("--known_percent",
                        type=int,
                        default=0,
                        help="Percent of supervised mixtures")
    parser.add_argument("--comment",
                        type=str,
                        default="",
                        help="Comment for current experiment")
    parser.add_argument("--log_dir",
                        type=str,
                        default="",
                        help="Log_dir for current experiment")
    args = parser.parse_args()
    logger.info("Arguments in command:\n{}".format(pprint.pformat(vars(args))))
    
    run(args)
