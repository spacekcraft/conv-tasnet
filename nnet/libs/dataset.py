# wujian@2018
# xpavlu10@2020

import random
import torch as th
import numpy as np
from typing import Dict, List, Tuple

from torch.utils.data.dataloader import default_collate
import torch.utils.data as dat

from .audio import WaveReader, DIHARDReader

import re

import pdb

def make_dataloader(train=True,
                    data_kwargs=None,
                    num_workers=4,
                    chunk_size=32000,
                    batch_size=16,
                    mixofmix = False):
    if mixofmix is False:
        dataset = Dataset(**data_kwargs)
    else:
        dataset = MixOfMixDataset(**data_kwargs)
    return DataLoader(dataset,
                      train=train,
                      chunk_size=chunk_size,
                      batch_size=batch_size,
                      num_workers=num_workers,
                      mixofmix=mixofmix)


class Dataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp="", ref_scp=None, sample_rate=8000):
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [
            WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp
        ]

    def __len__(self):
        return len(self.mix)

    def __getitem__(self, index):
        key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = [reader[key] for reader in self.ref]
        return {
            "mix": mix.astype(np.float32),
            "ref": [r.astype(np.float32) for r in ref],
        }

class MixOfMixDataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, mix_scp:str="", ref_scp:List=None, sample_rate:int=8000, knownPercent: int = 0, mix_supervised: bool = False, only_supervised: bool = False):
        """[summary]

        Args:
            mix_scp (str, optional): [description]. Defaults to "".
            ref_scp (List, optional): [description]. Defaults to None.
            sample_rate (int, optional): [description]. Defaults to 8000.
            knownPercent (int, optional): [description]. Defaults to 0.
            mix_supervised (bool, optional): If set to True then the supervised part of dataset is also mixed. Defaults to False.
            only_supervised (bool, optional): If set to True then only supervised part from dataset is provided. Defaults to False.
        """        
        self.only_supervised = only_supervised
        print("Only supervised?: {}".format(self.only_supervised))
        self.mix_supervised = mix_supervised
        self.knownPercent = knownPercent
        self.mix = WaveReader(mix_scp, sample_rate=sample_rate)
        self.ref = [
            WaveReader(ref, sample_rate=sample_rate) for ref in ref_scp
        ]
        
        #select unknown keys and remember which is unknown and which known
        self.unknownMixKeys = []
        self.knownMixKeys = []
        for key in self.mix.index_keys:
            #throw unfair coin
            if  random.random() > (self.knownPercent/100): #if uknown
                self.unknownMixKeys.append(key)        
            else:
                self.knownMixKeys.append(key)

    def _mix_mixture_of_mixtures(self, key:str, mix:np.ndarray, keysList:List)->Tuple[np.ndarray,str,np.ndarray,str,np.ndarray]:
        #get first and second speaker ID
        firstSpeaker = key[:3]
        secondSpeaker = key.split('_')[2][:3]
        #get aviable keys - not contains first or second speaker from the first mix
        aviableKeys = [key for key in keysList if not re.search("((("+firstSpeaker+")|("+secondSpeaker+"))?.*_.*_(("+firstSpeaker+")|("+secondSpeaker+")).*)|((("+firstSpeaker+")|("+secondSpeaker+")).*_.*_(("+firstSpeaker+")|("+secondSpeaker+"))?.*)", key)] 
        #randomly choose second key from aviable keys
        secondKey = aviableKeys[random.randint(0,len(aviableKeys)-1)]
        #get second mix
        secondMix = self.mix[secondKey] 
        #pad or cut
        if len(mix) < len(secondMix):
            secondMix = secondMix[:len(mix)]
        else:
            secondMix = np.pad(secondMix, (0,len(mix)-len(secondMix)), "constant",constant_values=(0,0))
        #mix mixtures
        return mix, key, secondMix, secondKey, mix+secondMix

    def __len__(self)->int:
        # if only supervised part
        if self.only_supervised:
            # return length of knwon mixture keys
            return len(self.knownMixKeys)
        # else return length of all mixtures
        return len(self.mix)

    def __getitem__(self, index:int)->Dict:
        # if only supervised is set, then get mixture key from the list of the supervised keys (known keys)
        if self.only_supervised:
            key = self.knownMixKeys[index]
        # else get key from all keys
        else:
            key = self.mix.index_keys[index]
        mix = self.mix[key]
        ref = [reader[key] for reader in self.ref] # read references
        
        #decide if known or uknown
        known = False if key in self.unknownMixKeys else True
        
        #if known get references
        if known is True:
            if self.mix_supervised:
                _, _, secondMix, secondKey, mixed = self._mix_mixture_of_mixtures(key, mix, self.knownMixKeys)
                ref += [reader[secondKey] for reader in self.ref] # read references for the second key
                
                return {
                    "mix": mixed.astype(np.float32),
                    "ref": [r.astype(np.float32) for r in ref],
                    "known": known # this to tell if it is known source signals or not 
                }
            else:
                return {
                    "mix": mix.astype(np.float32),
                    "ref": [r.astype(np.float32) for r in ref],
                    "known": known # this to tell if it is known source signals or not 
                }
        #if uknown mix two mixtures and take mixtures as references
        else:
            _, _, secondMix, _, mixed = self._mix_mixture_of_mixtures(key, mix, self.unknownMixKeys)
            #get mixtures as refs
            return {
                "mix": mixed.astype(np.float32),
                "ref": [mix, secondMix],
                "known": known # this to tell if it is known source signals or not 
            }

class DIHARDWsjMixOfMixDataset(object):
    """
    Per Utterance Loader
    """
    def __init__(self, dihard_json:str, wsj_mix_scp:str="", wsj_ref_scp:List[str]=None, sample_rate:int=8000, knownPercent:int = 0):
        # knownPercent set percentage of supervised part of the dataset
        self.knownPercent = knownPercent
        self.wsj_mix = WaveReader(wsj_mix_scp, sample_rate=sample_rate)
        self.wsj_ref = [
            WaveReader(ref, sample_rate=sample_rate) for ref in wsj_ref_scp
        ]
        
        self.dihard = DIHARDReader(json_path = dihard_json, sample_rate = sample_rate, preload = 4)

        #compute number of supervised mixtures
        numberOfKnowns = int(len(self.dihard)/(100-self.knownPercent))*knownPercent
        #randomly select supervised mixtures from the wall street journal dataset
        self.knownMixKeys = random.sample(self.wsj_mix.keys(), numberOfKnowns)

    def _mix_mixture_of_mixtures(self, id:int)->Tuple[np.ndarray,np.ndarray,np.ndarray]:
        # choose second mix
        secondID = id
        # while the secondID is randomly choosed as the different than id and cut is not from the same mixture
        while (secondID == id) or (self.dihard.key(id) == self.dihard.key(secondID)):
            # generate random ID
            secondID = random.randint(0,len(self.dihard)-1)
        # get mixtures
        firstMix = self.dihard[id]
        secondMix = self.dihard[secondID]
        #cut or pad
        if len(firstMix) < len(secondMix):
            secondMix = secondMix[:len(firstMix)]
        else:
            secondMix = np.pad(secondMix, (0,len(firstMix)-len(secondMix)), "constant",constant_values=(0,0))
        # return
        return firstMix, secondMix, firstMix+secondMix

    def __len__(self)->int:
        return len(self.knownMixKeys)+len(self.dihard)

    def __getitem__(self, index:int)->Dict:
        # if unsupervised index
        if index < len(self.dihard):
            firstMix, secondMix, mom = self._mix_mixture_of_mixtures(index)
            ref = [firstMix, secondMix]
            return{
                "mix": mom.astype(np.float32),
                "ref": [r.astype(np.float32) for r in ref],
                "known": False # this to tell if it is known source signals or not 
            }
        # if supervised
        else:
            key = self.knownMixKeys[index-len(self.dihard)]
            mix = self.wsj_mix[key]
            ref = [reader[key] for reader in self.wsj_ref] # read references
            return{
                "mix": mix.astype(np.float32),
                "ref": [r.astype(np.float32) for r in ref],
                "known": True # this to tell if it is known source signals or not 
            }

class ChunkSplitter(object):
    """
    Split utterance into small chunks
    """
    def __init__(self, chunk_size, train=True, least=16000, mixofmix = False):
        self.chunk_size = chunk_size
        self.least = least
        self.train = train
        self.mixofmix = mixofmix

    def _make_chunk(self, eg, s):
        """
        Make a chunk instance, which contains:
            "mix": ndarray,
            "ref": [ndarray...]
        """
        chunk = dict()
        chunk["mix"] = eg["mix"][s:s + self.chunk_size]
        chunk["ref"] = [ref[s:s + self.chunk_size] for ref in eg["ref"]]
        if self.mixofmix is True:
            chunk["known"] = eg["known"]
        return chunk

    def split(self, eg):
        N = eg["mix"].size
        # too short, throw away
        if N < self.least:
            return []
        chunks = []
        # padding zeros
        if N < self.chunk_size:
            P = self.chunk_size - N
            chunk = dict()
            chunk["mix"] = np.pad(eg["mix"], (0, P), "constant")
            chunk["ref"] = [
                np.pad(ref, (0, P), "constant") for ref in eg["ref"]
            ]
            if self.mixofmix is True:
                chunk["known"] = eg["known"]
            chunks.append(chunk)
        else:
            # random select start point for training
            s = random.randint(0, N % self.least) if self.train else 0
            while True:
                if s + self.chunk_size > N:
                    break
                chunk = self._make_chunk(eg, s)
                chunks.append(chunk)
                s += self.least
        return chunks

class DataLoader(object):
    """
    Online dataloader for chunk-level PIT
    """
    def __init__(self,
                 dataset,
                 num_workers=4,
                 chunk_size=32000,
                 batch_size=16,
                 train=True,
                 mixofmix = False):
        self.batch_size = batch_size
        self.train = train
        self.splitter = ChunkSplitter(chunk_size,
                                      train=train,
                                      least=chunk_size // 2,
                                      mixofmix=mixofmix)
        # just return batch of egs, support multiple workers
        self.eg_loader = dat.DataLoader(dataset,
                                        batch_size=batch_size // 2,
                                        num_workers=num_workers,
                                        shuffle=train,
                                        collate_fn=self._collate)

    def _collate(self, batch):
        """
        Online split utterances
        """
        chunk = []
        for eg in batch:
            chunk += self.splitter.split(eg)
        return chunk

    def _merge(self, chunk_list):
        """
        Merge chunk list into mini-batch
        """
        N = len(chunk_list)
        if self.train:
            random.shuffle(chunk_list)
        blist = []
        for s in range(0, N - self.batch_size + 1, self.batch_size):
            batch = default_collate(chunk_list[s:s + self.batch_size])
            blist.append(batch)
        rn = N % self.batch_size
        return blist, chunk_list[-rn:] if rn else []

    def __iter__(self):
        chunk_list = []
        for chunks in self.eg_loader:
            chunk_list += chunks
            batch, chunk_list = self._merge(chunk_list)
            for obj in batch:
                yield obj


