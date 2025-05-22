import random
import torch 
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd 
import numpy as np
from pathlib import Path


if torch.backends.mps.is_available(): 
    BASE_PATH="/Users/kmc479/Desktop/DCASE25"
         # Apple Silicon GPU
elif torch.cuda.is_available():
    BASE_PATH="/projappl/project_462000765/casciott/DCASE25"
class SoundDS(Dataset):
    def __init__(self, df, data_path, pre_aug=False):
        self.df = df
        self.data_path = str(data_path)

        self.pre_aug = pre_aug

        data = []
        targets = []
        device = "cpu"
        for i in range(len(df)):
            data.append(self.__getitem__(i)[0])
            targets.append(self.__getitem__(i)[1])
        #print(data[0])
        self.data = torch.stack(data, dim=0)
        print(self.data.shape)
        # print(self.data.shape)
        # print(type(self.data))
        # print(type(self.data. __getitem__(0)[0]))
        # print(self.data. __getitem__(0).shape)
        

        self.targets = targets
        data = np.array(data)
        self.data = torch.tensor(data, device=device)
        
    def __getitem__(self, index):
        sig, sr = torchaudio.load(f"{self.data_path}/audio/{self.df.loc[index, 'filename']}")
        if self.pre_aug: 
            sig, sr = self.time_shift(sig, sr)
        spectro = self.spectro_gram((sig, sr), 128, 2048, 512)
        
        return spectro, self.df.loc[index, "target"]
    
    def __len__(self):
        return len(self.df)
    
    def time_shift(self, sig, sr, shift_limit=2):
        _, sig_len = sig.shape
        shift = int(random.random*sig_len*shift_limit)
        return sig.roll(shift), sr

    def spectro_gram(self, aud, n_mels, n_fft, hop_len):
        sig,sr = aud
        top_db = 80

        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = torchaudio.transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
def get_split(dataset, data_path, val_fold=2):
    # train_df, val_df = train_test_split(
    #     meta[meta["fold"] != 5],
    #     test_size=0.20,            # 20 % of the *training* folds
    #     stratify=meta.loc[meta["fold"] != 5, "category"],
    #     random_state=42)
    
    return SoundDS(dataset[dataset["fold"]!=val_fold].reset_index(drop=True), data_path), SoundDS(dataset[dataset["fold"]==val_fold].reset_index(drop=True), data_path)

#read the metadata file
if torch.backends.mps.is_available(): 
    BASE_PATH="/Users/kmc479/Desktop/DCASE25"
         # Apple Silicon GPU
else:
    BASE_PATH="/projappl/project_462000765/casciott/DCASE25"
fd = pd.read_csv(f"{BASE_PATH}/SoftHebb-main/Training/data/ESC-50-master/meta/esc50.csv")
fd = fd[["fold", "target", "filename"]]
train, validation = get_split(fd, data_path=f"{BASE_PATH}/SoftHebb-main/Training/data/ESC-50-master")


print(validation.data.shape)

