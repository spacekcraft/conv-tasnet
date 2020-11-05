#!/usr/bin/env python

# wujian@2018
"""
Compute SI-tqdSDR as the evaluation metric
"""

import argparse

from tqdm import tqdm

from collections import defaultdict
from libs.metric import si_snr, permute_si_snr, permute_si_snr_mix_of_mix
from libs.audio import WaveReader, Reader

from glob import glob
import numpy as np

import matplotlib.pyplot as plt

import pickle

import pdb

class SpeakersReader(object):
    def __init__(self, scps):
        split_scps = scps.split(",")
        if len(split_scps) == 1:
            raise RuntimeError(
                "Construct SpeakersReader need more than one script, got {}".
                format(scps))
        self.readers = [WaveReader(scp) for scp in split_scps]

    def __len__(self):
        first_reader = self.readers[0]
        return len(first_reader)

    def __getitem__(self, key):
        return [reader[key] for reader in self.readers]

    def __iter__(self):
        first_reader = self.readers[0]
        for key in first_reader.index_keys:
            yield key, self[key]


class Report(object):
    def __init__(self, spk2gender=None, outputDir = None):
        self.s2g = Reader(spk2gender) if spk2gender else None
        self.snr = defaultdict(float)
        self.cnt = defaultdict(int)
        self.outputDir = outputDir
        self.snrList = []
    def add(self, key, val):
        gender = "NG"
        if self.s2g:
            gender = self.s2g[key]
        self.snr[gender] += val
        self.cnt[gender] += 1
        self.snrList.append(val)

    def report(self):
        print("SI-SDR(dB) Report: ")
        for gender in self.snr:
            tot_snrs = self.snr[gender]
            num_utts = self.cnt[gender]
            print("{}: {:d}/{:.3f}".format(gender, num_utts,
                                           tot_snrs / num_utts))
            if self.outputDir:
                with open("{}/evalResult_{:.3f}.log".format(self.outputDir, tot_snrs/num_utts), "w") as outFile:
                    outFile.write("{}: {:d}/{:.3f} SI-SNR".format(gender, num_utts,
                                           tot_snrs / num_utts))
                plt.figure()
                plt.title("Histogram\n{}: {:d}/{:.3f} SI-SNR".format(gender, num_utts,
                                           tot_snrs / num_utts))
                plt.hist(self.snrList, bins = 30)
                plt.savefig(self.outputDir+"/histogram.png")

                with open(self.outputDir+"/snr.pkl",'wb') as f:
                    pickle.dump(self.snrList, f)


                

#xpavlu10 edit
def run(args):
    print("Working on folder {}".format(args.sep_scp))
    #get all scp files from separation folder
    folder = sorted(glob(args.sep_scp+'/*.scp', recursive=False)) # ndarray of names of all samples
    sep_scp = ""
    for scp in folder: # build string from array of scp files
        sep_scp += scp + "," 
    sep_scp = sep_scp[:-1] #remove last comma
    
    single_speaker = len(sep_scp.split(",")) == 1
    reporter = Report(args.spk2gender, outputDir=args.sep_scp)

    if single_speaker:
        sep_reader = WaveReader(sep_scp)
        ref_reader = WaveReader(args.ref_scp)
        for key, sep in tqdm(sep_reader):
            ref = ref_reader[key]
            if sep.size != ref.size:
                end = min(sep.size, ref.size)
                sep = sep[:end]
                ref = ref[:end]
            snr = si_snr(sep, ref)
            reporter.add(key, snr)
    else:
        sep_reader = SpeakersReader(sep_scp)
        ref_reader = SpeakersReader(args.ref_scp)

        for key, sep_list in tqdm(sep_reader):
            ref_list = ref_reader[key]
            zero_ref_list = ref_reader[key]
            if args.mixofmix != 0:
                if len(ref_list) > len(sep_list):
                    raise RuntimeError("There are more references then separs")
                #create zero references
                for i in range(len(sep_list) - len(ref_list)):
                    zero_ref_list.append(np.zeros_like(ref_list[0])+0.0001)
            #Cut lengths
            if sep_list[0].size != ref_list[0].size:
                end = min(sep_list[0].size, ref_list[0].size)
                sep_list = [s[:end] for s in sep_list]
                ref_list = [s[:end] for s in ref_list]
                zero_ref_list = [s[:end] for s in zero_ref_list]
            if args.mixofmix != 0: # get right outputs combination
                right_sep_list = permute_si_snr_mix_of_mix(sep_list, zero_ref_list)  
            else: right_sep_list = sep_list #compatibility
            #PIT
            snr = permute_si_snr(right_sep_list[:len(ref_list)], ref_list)
            reporter.add(key, snr)
            #reporter.report()
    reporter.report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=
        "Command to compute SI-SDR, as metric of the separation quality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "sep_scp",
        type=str,
        help="Separated speech scripts, waiting for measure and also results output"
        "(support multi-speaker, egs: spk1.scp,spk2.scp)")
    parser.add_argument(
        "ref_scp",
        type=str,
        help="Reference speech scripts, as ground truth for"
        " SI-SDR computation")
    parser.add_argument(
        "--spk2gender",
        type=str,
        default="",
        help="If assigned, report results per gender")
    parser.add_argument(
        "--mixofmix",
        type=int,
        default=0,
        help="If not zero, use mixtures of mixtures evaluation")
    args = parser.parse_args()
    run(args)