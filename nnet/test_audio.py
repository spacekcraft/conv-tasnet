from tqdm import tqdm

from libs import audio
from libs import dataset
import pdb

if __name__ == "__main__":
    #reader = audio.DIHARDReader(json_path = "/pub/users/xpavlu10/DIHARD_calls/cuts.json", sample_rate = 16000)
    #for i, samples in tqdm(reader):
    #    pass

    dwdataset = dataset.DIHARDWsjMixOfMixDataset(dihard_json = "/pub/users/xpavlu10/DIHARD_calls/cuts.json", 
                                                 wsj_mix_scp = "/pub/users/xpavlu10/min_dataset/tr/mix.scp", 
                                                 wsj_ref_scp=["/pub/users/xpavlu10/min_dataset/tr/s1.scp","/pub/users/xpavlu10/min_dataset/tr/s2.scp"],
                                                 knownPercent=20)

    for i in tqdm(dwdataset):
        pass