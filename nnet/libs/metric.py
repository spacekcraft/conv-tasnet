# wujian@2018
"""
SI-SNR(scale-invariant SNR/SDR) measure of speech separation
"""

import numpy as np

from itertools import permutations

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
