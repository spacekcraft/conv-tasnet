import glob
import json
import tqdm

from .audio import read_wav

import pdb

class PreProcessDIHARD(object):
    """PreProcess class is initialized with the parameters of seconds per path and second of overlap
    It loads wavs from the file and creates json file, where 

    Args:
        object ([type]): [description]
    """    
    def __init__(self, secPart = 6, overlap = 0):
        self.secPart = secPart
        self.overlap = overlap
    
    def preprocess(self, wav_path, scp_path):
        cuts = list()
        for file in tqdm.tqdm(glob.glob("{}/*".format(wav_path))):
            #load file
            samp_rate, samps = read_wav(file, normalize=False, return_rate=True)
            #get length
            lengthSeconds = len(samps)/samp_rate
            #get cuts
            numberOfParts = (int) (lengthSeconds/self.secPart)
            #for cuts in file:
            for i in range(numberOfParts):
                stop = ((i+1)*self.secPart)*samp_rate
                # control end of file
                if stop > len(samps):
                    stop = len(samps)
                filePart = dict()
                filePart["name"] = file
                filePart["start"] = (i*self.secPart)*samp_rate
                filePart["stop"] = stop
                filePart["part"] = i
                filePart["total_parts"] = numberOfParts
                cuts.append(filePart)
        #write to json dict
        with open(scp_path, "w") as outJson:
            outJson.write(json.dumps(cuts,indent=4))

if __name__ == "__main__":
    preprocessor = PreProcessDIHARD(secPart = 6, overlap = 0)
    preprocessor.preprocess("/pub/users/xpavlu10/DIHARD_calls/wavs", "/pub/users/xpavlu10/DIHARD_calls/cuts.json")