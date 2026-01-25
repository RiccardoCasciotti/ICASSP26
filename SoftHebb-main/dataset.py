import copy
import random
import h5py
try:
    from utils import seed_init_fn, DATASET
except:
    from hebb.utils import seed_init_fn, DATASET
import numpy as np
import os
import os.path as op
import torch
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, FashionMNIST, STL10, ImageNet, ImageFolder
from typing import Optional, Any
from utils import load_presets, get_device
import torchaudio
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import pandas as pd 
from pathlib import Path

torch.cuda.empty_cache()
if torch.backends.mps.is_available(): 
    BASE_PATH="/Users/kmc479/Desktop/ICASSP26/SoftHebb-main"
         # Apple Silicon GPU
elif torch.cuda.is_available():
    BASE_PATH="/scratch/project_462001198/casciott"

class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size(), device=tensor.device) * self.std + self.mean



def select_dataset(dataset_config, device, dataset_path):
    test_transform = None
    val_indices = None

    split = dataset_config["split"] if "split" in dataset_config else "train"
    
    
    if dataset_config['name'] == 'ESC50':
        dataset_class = ESC50
        indices = list(range(2000))
        dataset_path = f"{BASE_PATH}/Training/data/ESC-50-master"
        # if dataset_config['augmentation']:
        #     dataset_train_class = AugFastCIFAR100
        #     dataset_config['num_workers'] = 4
        #     device = 'cpu'
        #     transform = crop_flip(dataset_config['width'], dataset_config['height'])
    
        dataset_train_class = ESC50
        transform = None

   

    
    dataset_config['num_workers'] = 64
    return dataset_train_class, dataset_class, test_transform, transform, device, split, dataset_path

def get_indices(dataset_config, indices):
    indices = list(range(indices))
    train_indices  = None
    val_indices = None
    if not isinstance(dataset_config['training_class'], str):
        # we have to select indices up to the training_sample (trainign set size) otherwise the future origin_dataset
        # won't have enough indeces (it only stores the datapoints of the chosen training_class(es)
        # Another headache is that although you give training_sample, the validation set is taken from that
        # In the end: if want to validate, do it with only the same class(es)
        not_all_classes_samples = dataset_config['training_sample']
        if dataset_config['validation']:
            not_all_classes_samples += dataset_config['val_sample']
        not_all_classes_indices = indices[:not_all_classes_samples]
        indices = copy.deepcopy(not_all_classes_indices)

    if dataset_config['shuffle']:
        np.random.shuffle(indices)
        train_indices = indices[:dataset_config['training_sample']]

    if dataset_config['validation']:
        val_indices = indices[:dataset_config['val_sample']]
        train_indices = indices[
                        dataset_config['val_sample']:(dataset_config['training_sample'] + dataset_config['val_sample'])]
    return  train_indices, val_indices

def reshape_dataset(dataset, old_size):
    new_data = torch.Tensor((dataset.data).shape[0], (dataset.data).shape[1], old_size, old_size )
    for i in range(((dataset.data).shape)[0]):
        new_data[i] = transforms.Resize(old_size)(dataset.data[i])
    dataset.data = new_data
    return dataset

def get_split(dataset, data_path, dataset_config, test_fold):
    # train_df, val_df = train_test_split(
    #     meta[meta["fold"] != 5],
    #     test_size=0.20,            # 20 % of the *training* folds
    #     stratify=meta.loc[meta["fold"] != 5, "category"],
    #     random_state=42)
    val_fold = (test_fold + 1)%5 +1
    print("VAL_FOLD TEST_FOLD: ", val_fold, test_fold)
    folds = [val_fold, test_fold]
    
    return ESC50(dataset[~dataset["fold"].isin(folds)].reset_index(drop=True), data_path, dataset_config, augment=False), ESC50(dataset[dataset["fold"]==val_fold].reset_index(drop=True), data_path, dataset_config, augment=False), ESC50(dataset[dataset["fold"]==test_fold].reset_index(drop=True), data_path, dataset_config, augment=False)

def make_data_loaders(dataset_config, batch_size, device, dataset_path=DATASET):
    """
     Load Mnist Dataset and create a dataloader

    Parameters
    ----------
    dataset_config : dict
        Configuration of the expected dataset
    batch_size: int
    dataset_path : str path
        Path to the dataset folder.

    Returns
    -------
    train_loader : torch.utils.data.DataLoader
        Training dataloader.
    test_loader : torch.utils.data.DataLoader
        Testing dataloader.

    """
    g = torch.Generator()
    if dataset_config['seed'] is not None:
        seed_init_fn(dataset_config['seed'])
        g.manual_seed(dataset_config['seed'] % 2 ** 32)

    if dataset_config["name"] == "ESC50":
        classes_offset = []
        fd = pd.read_csv(f"{BASE_PATH}/Training/data/ESC-50-master/meta/esc50.csv")
        fd = fd[["fold", "target", "filename"]]
        train_split, val_split, test_split = get_split(fd, dataset_config=dataset_config, data_path=f"{BASE_PATH}/Training/data/ESC-50-master", test_fold=dataset_config["fold"])
        if "n_classes" in dataset_config:
            selected_classes = dataset_config["selected_classes"]
            if dataset_config["shmh"] or dataset_config["SINGLE"]:
                test_split, classes_offset = classes_subset(dataset_config, test_split, selected_classes, device, False) 
            else: 
                test_split, classes_offset = classes_subset(dataset_config, test_split, selected_classes, device, True)
            if dataset_config["SINGLE"]:
                train_split = classes_subset(dataset_config, train_split, selected_classes, device, False)
                val_split = classes_subset(dataset_config, val_split, selected_classes, device, False)
            else:
                train_split, classes_offset = classes_subset(dataset_config, train_split, selected_classes, device, True)
                val_split, _ = classes_subset(dataset_config, val_split, selected_classes, device, True)
    elif dataset_config["name"] == "URBANSOUND8K":

        selected_classes = dataset_config["selected_classes"]
        eval_fold = dataset_config["fold"]
        data_train = Urbansound8k(data_path=f"/scratch/project_462001198/casciott/datasets/urbansound8k", selected_classes=selected_classes, test=False, eval_fold=eval_fold, debug=False)
        data_train, classes_offset= class_cleaner(dataset_config, data_train, selected_classes)
        train_split, val_split = torch.utils.data.random_split(data_train, [0.9, 0.1])
        test_split = Urbansound8k(data_path=f"/scratch/project_462001198/casciott/datasets/urbansound8k", selected_classes=selected_classes, test=True, eval_fold=eval_fold, debug=False)
        test_split, classes_offset = class_cleaner(dataset_config, test_split, selected_classes)

    train_loader = torch.utils.data.DataLoader(dataset=train_split,
                                            batch_size=batch_size,
                                            num_workers=dataset_config['num_workers'],
                                                
    
    )
    val_loader = torch.utils.data.DataLoader(dataset=val_split,
                                            batch_size=batch_size,
                                            num_workers=dataset_config['num_workers'],
                                                
    
    )
    test_loader = torch.utils.data.DataLoader(dataset=test_split,
                                                batch_size=batch_size,
                                                num_workers=dataset_config['num_workers'],
                                                )
    print(classes_offset)
    return train_loader, val_loader, test_loader, classes_offset

def class_cleaner(dataset_config, dataset, selected_classes):
# Cleans the classes so that it guarantees that there is first class with index 0 in the dataset, 
# since it is required by torch. 

    
    if dataset_config["name"] == "ESC50":
        targets = dataset.targets

    elif dataset_config["name"] == "URBANSOUND8K":
        targets = dataset.targets[:,0]

    
    selected_classes.sort()

    classes_offset = [] 
    for i in range(len(selected_classes)): 
        for j in range(len(targets)): 
            #if  targets[j] == selected_classes[i]: 
                # 
            #    targets[j] =  i
            targets[targets==selected_classes[i]] = i
        classes_offset.append(selected_classes[i]-i)
    if dataset_config["name"] == "ESC50" or dataset_config["name"] == "URBANSOUND8K":
        dataset.targets = torch.tensor(targets, device=get_device(), dtype=torch.long)
    

        

    #if 0 not in selected_classes: 
    #     
    #    min_value = min(dataset.targets)
    #    dataset.targets = dataset.targets - min_value # filter doesn't work in this case
    # 
    # 
    # 
    # 
    
    # if dataset_config["name"] == "STL10":
        # print("TARGETS AFTER CLEANER: ", dataset.labels[:20])
    # else:
        # print("TARGETS AFTER CLEANER: ", dataset.targets[:20])
        # 
        # 
        # 

         
         
         
    # print("------------------------")
    

    return dataset, classes_offset

def classes_subset(dataset_config, dataset,selected_classes, device, class_clean=True):
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # I don't think it will work with ImageNette 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Creates a dataset made up of a subsets of classes indicated in the selected classes variable.
    

    if dataset_config["name"] == "ESC50": 
        T = dataset.targets.cpu().numpy()


    classes = torch.tensor(selected_classes)
    indices = (torch.tensor(T)[..., None] == classes).any(-1).nonzero(as_tuple=True)[0]
    indices = indices.tolist()
    T = list(T[indices])
     
    D = dataset.data.detach().cpu().numpy()
    D = list(D[indices])
    dataset.data = D
    dataset.data = torch.tensor(dataset.data, device=get_device())
   
    


    
    if dataset_config["name"] == "ESC50": 
        dataset.targets = torch.tensor(T, device=get_device())
   
    classes_offset = []
    if class_clean:
        dataset, classes_offset = class_cleaner(dataset_config ,dataset, selected_classes)


    return dataset, classes_offset

# *************************************************** ESC50 ***************************************************

class ESC50(Dataset):
    def __init__(self, df, data_path, dataset_config, augment):
        self.df = df.sample(frac=1).reset_index(drop=True)
        
        self.data_path = data_path
        data = []
        targets = []
        device = get_device()
        for i in range(len(df)):
            sig, sr = torchaudio.load(f"{self.data_path}/audio/{self.df.loc[i, 'filename']}")
            data.append(self.spectro_gram((sig, sr), n_mels=dataset_config["n_mels"], n_fft=dataset_config["n_fft"], hop_len=dataset_config["hop_len"], augment=augment))
            targets.append(self.df.loc[i, "target"])
        

        self.data = torch.stack(data, dim=0)
        shape = self.data.shape
        #self.data = torch.Tensor.reshape(self.data, (shape[0], shape[2], shape[3], shape[1]))
        self.data = np.array(data)
        self.targets = np.array(targets)
        self.data = torch.tensor(self.data, device=device)
        self.targets = torch.tensor(self.targets, device=device)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
    
    def time_shift(self, signal):
        sig, sr = signal
        sig_len = len(sig)
        shift = int(np.random.random()*sig_len*0.3)
        return (torch.tensor(np.roll(sig, shift)), sr)
    def __len__(self):
        return len(self.data)

    def spectro_gram(self, aud, n_mels, n_fft, hop_len, augment):
        sig,sr = aud
        top_db = 80
        # if augment: 
        #     sig, sr = self.time_shift(aud)
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = torchaudio.transforms.AmplitudeToDB(top_db=80)(spec)
        if augment: 
            spec = torchaudio.transforms.FrequencyMasking(freq_mask_param=10)(spec)
            spec = torchaudio.transforms.TimeMasking(time_mask_param=10)(spec)

        return (spec)

# *************************************************** URBANSOUND8K ***************************************************
LABEL2ID_Urbansound8k = {0: "air_conditioner",
1 : "scar_horn",
2: "children_playing",
3:"dog_bark",
4 : "drilling",
5 : "engine_idling",
6 : "gun_shot",
7 : "jackhammer",
8 : "siren",
9 : "street_music"}

class Urbansound8k(Dataset):
    def __init__(self, data_path, selected_classes=list(LABEL2ID_Urbansound8k.keys()), test=False, eval_fold=0, debug=False):
        self.dataset = h5py.File(f'{data_path}/h5s/urbansound8k.h5', "r")
        self.selected_classes=sorted(selected_classes)
        self.task_indexes = self.__get_task_dataset_indexes_from_hd5__(test, eval_fold, debug)

        np.random.shuffle(self.task_indexes)
        self.data = self.dataset["mel_spectrogram"]
        self.targets = self.dataset["labels_ids"]
        
        
        self.start = 0
        self.end = len(self.task_indexes)

        # self.pos_weights = self.__class_imbalance_weights__()

    def __len__(self):
        return self.end-self.start
    
    def __getitem__(self, index):
        t = self.targets[self.task_indexes[self.start+index]]
        return torch.from_numpy(self.data[self.task_indexes[self.start+index]]).t().unsqueeze(0), t

    def __class_imbalance_weights__(self):
        N = self.__len__()
        labels = np.stack([ self.__getitem__(i)[1] for i in range(N) ])  
        pos_counts = labels.sum(axis=0)                                
        neg_counts = N - pos_counts                                     

        pos_counts = np.where(pos_counts == 0, 1, pos_counts)
        pos_weight = neg_counts / pos_counts                            
        return torch.from_numpy(pos_weight)

    
    def __get_task_dataset_indexes_from_hd5__(self, test=False, eval_fold =0, debug=False):

        data = []
        
        if debug:
            length = 2000
        else:
            length = len(self.dataset["filenames"])
        for i in range(length):
            entry = self.dataset["one_hot_labels"][i]
            res = np.any(entry[self.selected_classes])
            fold_check = self.dataset["folds"][i] == eval_fold
            if not test and res and not fold_check: 
                data.append(i)
            elif test and res and fold_check:
                data.append(i)

        return data    

    def __del__(self):
        try:
            if hasattr(self, "dataset") and self.dataset:
                self.dataset.close()
        except Exception as e:
                pass
        
        