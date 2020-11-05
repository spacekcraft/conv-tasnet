import logging
from glob import glob

import pdb

class Dataset():
    def __init__(self, path, mixPath = "", speakersPath = [], sample_rate = 8000):
        self.path = path
        self.mix = sorted(glob(path+mixPath+'/*.wav', recursive=False)) # ndarray of names of all samples
        self.speakers = [sorted(glob(path+speakerPath+'/*.wav', recursive=False)) for speakerPath in speakersPath] # ndarray of ndarrays of speakers sample names
    def __len__(self):
        return len(self.mix)
    def __getitem__(self, idx):
        return {"mix": self.mix[idx], 
                "speakers": [speaker[idx] for speaker in self.speakers]}

def generate(path):
    print("Generating SCPs for path:", path)
    fmix = open(path+"mix.scp", 'w')
    fs1 = open(path+"spk1.scp", 'w')
    fs2 = open(path+"spk2.scp", 'w')
    dataset = Dataset(path, "mix", ["s1","s2"])
    for i in dataset:
        fmix.write(i["mix"].split('/')[-1].strip('.wav')+" "+i["mix"]+"\n")
        fs1.write(i["speakers"][0].split('/')[-1].strip('.wav')+" "+i["speakers"][0]+"\n")
        fs2.write(i["speakers"][1].split('/')[-1].strip('.wav')+" "+i["speakers"][1]+"\n")
    fmix.close()
    fs1.close()
    fs2.close()

def generateFile(pathToFolder, pathToFile):
    print("Generating SCP", pathToFile, "for path:", pathToFolder)
    fs = open(pathToFile, 'w')

    folder = sorted(glob(pathToFolder+'/*.wav', recursive=False)) # ndarray of names of all samples
    for i in folder:
        #print(i.split('/')[-1].strip('.wav')+" "+i+"\n")
        fs.write(i.split('/')[-1].strip('.wav')+" "+i+"\n")
    fs.close()
    print("Generated")


if __name__ == "__main__":
    #generate("../min_dataset/tr/")
    #generate("../min_dataset/cv/")
    #generate("../min_dataset/tt/")
    generateFile("./out/1/spk1", "./out/1/spk1.scp")
    generateFile("./out/1/spk2", "./out/1/spk2.scp")