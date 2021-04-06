# wujian@2018
"""
SI-SNR(scale-invariant SNR/SDR) measure of speech separation
"""

import numpy as np
import torch as th

from itertools import permutations
from itertools import combinations

import pdb

def si_snr(x, s, remove_dc=True):
    """
    Compute SI-SNR
    Arguments:
        x: vector, enhanced/separated signal
        s: vector, reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    # zero mean, seems do not hurt results
    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm)**2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s)**2
        n = x - t
    return 20 * np.log10((vec_l2norm(t) / (vec_l2norm(n)+0.000001))+0.000001)


def permute_si_snr(xlist, slist):
    """
    Compute SI-SNR between N pairs
    Arguments:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal(ground truth)
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrs = []
    for order in permutations(range(N)):
        si_snrs.append(si_snr_avg(xlist, [slist[n] for n in order]))
    return max(si_snrs)

def permute_si_snr_mix_of_mix(xlist, slist):
    """
    Find the best combination between N pairs depending on SI-SNR
    Arguments:
        x: list[vector], enhanced/separated signal
        s: list[vector], reference signal(ground truth)
    Return:
        order: list[vector], list of outputs in the best order
    """

    def si_snr_avg(xlist, slist):
        return sum([si_snr(x, s) for x, s in zip(xlist, slist)]) / len(xlist)

    N = len(xlist)
    if N != len(slist):
        raise RuntimeError(
            "size do not match between xlist and slist: {:d} vs {:d}".format(
                N, len(slist)))
    si_snrMem = None
    bestOrder=None
    for order in permutations(range(N)):
        new_si_snr = si_snr_avg([xlist[n] for n in order], slist)
        #print("newSi-snr", new_si_snr,"order", order)
        if si_snrMem is None:  #init
            si_snrMem = new_si_snr
            bestOrder = order
        elif si_snrMem < new_si_snr: #if better result, then remember combination
            si_snrMem = new_si_snr
            bestOrder = order

    return [xlist[n] for n in bestOrder]


def genCombinations(N): # tohle generovat tak, aby byla pro 4 jen 1/3 2/2
    """
    Arguments:
    N: number of outputs
    Return:
    all: generated combinations
    """
    all = []
    m = [i for i in range(N)]
    for i in range(int(N/2),N):
        #pdb.set_trace()
        left = list(combinations(m, i))
        for eachLeft in left:
            eachLeft = sorted(eachLeft) #sort eachLeft
            x = [i for i in m if i not in eachLeft]
            for eachRight in list(combinations(x,N-len(eachLeft))): # where N-len(eachLeft)
                newOne = (eachLeft, sorted(eachRight))
                newOne2 = (sorted(eachRight), eachLeft)
                #pdb.set_trace()
                if (newOne not in all) and (newOne2 not in all):   #duplicity check, check if this pair exists in all, this is possible because eachLeft and newOne are sorted
                    all.append(newOne)
    return all

def combine_si_snr_mix_of_mix(ests, refs):
    '''Mix of mix objective function'''
    num_spks = len(refs)
    num_ests = len(ests) # number of estimates
    loss = None
    combs = genCombinations(num_ests)
    #print("Combinations generated", combs)
    #for each combination in combs
    for comb in combs:
        #generate mix of outputs
        first = None # first mix of estimates
        for out in comb[0]:
            if first is None:
                first = ests[out]
            else:    
                first += ests[out]
        second = None # second mix of estimates
        for out in comb[1]:
            if second is None:
                second = ests[out]
            else:    
                second += ests[out]
        #try both combinations
        if first is None or second is None:
            raise RuntimeError("First or Second is None in loss function.")
        #firstLoss is array of size N -> means batch size
        firstLoss = (si_snr(first, refs[0]) + si_snr(second, refs[1]))/2 # u sissnr chci vetsi hodnotu, tudiz bud obratit znamenko, nebo udelat maximum a pak vratit minus hodnotu
        secondLoss = (si_snr(second, refs[0]) + si_snr(first, refs[1]))/2
        #compare loss from both permutation
        
        stackedLoss = np.stack([firstLoss, secondLoss])
        newLoss = np.max(stackedLoss, axis=0)
        #compare with losses from other combinations        
        if loss is None:
            loss = newLoss
        else:
            stackedLoss = np.stack([newLoss, loss])
            loss = np.max(stackedLoss, axis=0)
    #print("Loss this is a loss {}".format(loss))
    return loss
