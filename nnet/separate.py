#!/usr/bin/env python

# wujian@2018
# xpavlu10@2020

import os
import argparse
import shutil
import subprocess

import torch as th
import numpy as np

from conv_tas_net import ConvTasNet

from libs.utils import load_json, get_logger
from libs.audio import WaveReader, DIHARDReader, write_wav

import matplotlib.pyplot as plt
from glob import glob

from tqdm import tqdm
import re
import random

import pdb

logger = get_logger(__name__)

logging = False

class NnetComputer(object):
    def __init__(self, cpt_dir, gpuid):
        if gpuid >= 0:
            #Nvidia smi call
            freeGpu = subprocess.check_output('nvidia-smi -q | grep "Minor\|Processes"| grep "None" -B1 | tr -d " " | cut -d ":" -f2 | sed -n "1p"', shell=True)
            if len(freeGpu) == 0: # if gpu not aviable use cpu
                self.device = th.device("cpu")
            else: self.device = th.device('cuda:'+freeGpu.decode().strip())
        else:
            self.device = th.device("cpu")
        nnet = self._load_nnet(cpt_dir)
        self.nnet = nnet.to(self.device) if gpuid >= 0 else nnet
        # set eval model
        self.nnet.eval()

    def _load_nnet(self, cpt_dir):
        nnet_conf = load_json(cpt_dir, "mdl.json")
        nnet = ConvTasNet(**nnet_conf)
        cpt_fname = os.path.join(cpt_dir, "best.pt.tar")
        cpt = th.load(cpt_fname, map_location="cpu")
        nnet.load_state_dict(cpt["model_state_dict"])
        logger.info("Load checkpoint from {}, epoch {:d}".format(
            cpt_fname, cpt["epoch"]))
        return nnet

    def compute(self, samps):
        with th.no_grad():
            raw = th.tensor(samps, dtype=th.float32, device=self.device)
            sps = self.nnet(raw)
            sp_samps = [np.squeeze(s.detach().cpu().numpy()) for s in sps]
            return sp_samps

#By xpavlu10
def generateFile(pathToFolder, pathToFile):
    print("Generating SCP", pathToFile, "for path:", pathToFolder)
    fs = open(pathToFile, 'w')

    folder = sorted(glob(pathToFolder+'/*.wav', recursive=False)) # ndarray of names of all samples
    for i in folder:
        fs.write(i.split('/')[-1].strip('.wav')+" "+i+"\n")
    fs.close()
    print("Generated")

def plotOutputs(path, spks):
    if logging is True: logger.info("Plot output {}".format(path))
    #if dir doesn't exists then create
    fdir = os.path.dirname(path)
    if fdir and not os.path.exists(fdir):
        os.makedirs(fdir)
    
    #get Mix and Min
    max = 0
    min = 0
    for idx, samps in enumerate(spks):
        newMax = np.max(samps)
        if max < newMax: max = newMax
        newMin = np.min(samps)
        if min > newMin: min = newMin

    #plot
    plt.figure()
    sl = np.ceil(len(spks)/2)
    for idx, samps in enumerate(spks):
        plt.subplot(sl, 2, idx+1)
        plt.title("Speaker {}".format(idx+1))
        plt.ylim(min-10, max+10)
        plt.plot(samps)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

def cpyModelInfo(checkpoint, dump_dir):
    os.mkdir(dump_dir+"/model_info")
    files = glob(checkpoint+"/*.json")
    for f in files:
        shutil.copy(f, dump_dir+"/model_info/")

def run(args):
    os.mkdir(args.dump_dir)
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu)
    cpyModelInfo(args.checkpoint, args.dump_dir)
    lenGen = 0
    for key, mix_samps in tqdm(mix_input):
        if logging is True: logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps)
        norm = np.linalg.norm(mix_samps, np.inf)
        lenGen = len(spks)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))

            write_wav(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
        if args.plot != 0: plotOutputs(os.path.join(args.dump_dir, "plot_spk/{}.png".format(key)), spks)
    #generate SCP files
    for idx in range(lenGen):
        generateFile(os.path.join(args.dump_dir, "spk{}".format(
                    idx + 1)), os.path.join(args.dump_dir, "spk{}.scp".format(
                    idx + 1)))
    logger.info("Compute over {:d} utterances".format(len(mix_input)))

def run2(args):
    os.mkdir(args.dump_dir)
    mix_input = WaveReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu)
    cpyModelInfo(args.checkpoint, args.dump_dir)
    lenGen = 0
    for key, mix_samps in tqdm(mix_input):
        if logging is True: logger.info("Compute on utterance {}...".format(key))
        
        firstSpeaker = key[:3]
        secondSpeaker = key.split('_')[2][:3]

        aviableKeys = [key for key in mix_input.index_keys if not re.search("((("+firstSpeaker+")|("+secondSpeaker+"))?.*_.*_(("+firstSpeaker+")|("+secondSpeaker+")).*)|((("+firstSpeaker+")|("+secondSpeaker+")).*_.*_(("+firstSpeaker+")|("+secondSpeaker+"))?.*)", key)] 
            
        secondKey = aviableKeys[random.randint(0,len(aviableKeys)-1)]
        secondMix = mix_input[secondKey]

        if len(mix_samps) < len(secondMix):
                secondMix = secondMix[:len(mix_samps)]
        else:
            secondMix = np.pad(secondMix, (0,len(mix_samps)-len(secondMix)), "constant",constant_values=(0,0))

        mixofmix = mix_samps+secondMix

        spks = computer.compute(mixofmix)
        norm = np.linalg.norm(mixofmix, np.inf)
        lenGen = len(spks)
        for idx, samps in enumerate(spks):
            samps = samps[:mixofmix.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))

            write_wav(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
        if args.plot != 0: plotOutputs(os.path.join(args.dump_dir, "plot_spk/{}.png".format(key)), spks)
    #generate SCP files
    for idx in range(lenGen):
        generateFile(os.path.join(args.dump_dir, "spk{}".format(
                    idx + 1)), os.path.join(args.dump_dir, "spk{}.scp".format(
                    idx + 1)))
    logger.info("Compute over {:d} utterances".format(len(mix_input)))

def run_DIHARD(args):
    """Run separation on DIHARD dataset

    Args:
        args ([type]): [description]
    """    
    os.mkdir(args.dump_dir)
    mix_input = DIHARDReader(args.input, sample_rate=args.fs)
    computer = NnetComputer(args.checkpoint, args.gpu)
    cpyModelInfo(args.checkpoint, args.dump_dir)
    lenGen = 0
    for key, mix_samps in tqdm(mix_input):
        if logging is True: logger.info("Compute on utterance {}...".format(key))
        spks = computer.compute(mix_samps)
        norm = np.linalg.norm(mix_samps, np.inf)
        lenGen = len(spks)
        for idx, samps in enumerate(spks):
            samps = samps[:mix_samps.size]
            # norm
            samps = samps * norm / np.max(np.abs(samps))

            write_wav(
                os.path.join(args.dump_dir, "spk{}/{}.wav".format(
                    idx + 1, key)),
                samps,
                fs=args.fs)
        if args.plot != 0: plotOutputs(os.path.join(args.dump_dir, "plot_spk/{}.png".format(key)), spks)
    #generate SCP files
    for idx in range(lenGen):
        generateFile(os.path.join(args.dump_dir, "spk{}".format(
                    idx + 1)), os.path.join(args.dump_dir, "spk{}.scp".format(
                    idx + 1)))
    logger.info("Compute over {:d} utterances".format(len(mix_input)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to do speech separation in time domain using ConvTasNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("checkpoint", type=str, help="Directory of checkpoint")
    parser.add_argument(
        "--input", type=str, required=True, help="Script for input waveform")
    parser.add_argument(
        "--gpu",
        type=int,
        default=-1,
        help="GPU device to offload model to, -1 means running on CPU")
    parser.add_argument(
        "--fs", type=int, default=8000, help="Sample rate for mixture input")
    parser.add_argument(
        "--dump-dir",
        type=str,
        default="sps_tas",
        help="Directory to dump separated results out")
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="If not 0, then plot results")
    parser.add_argument(
        "--type", type=str, default="MOM", help="MOM|DIHARD")
    args = parser.parse_args()

    if args.type == "MOM":
        run(args)
    elif args.type == "DIHARD":
        run_DIHARD(args)