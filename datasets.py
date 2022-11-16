import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os

class ArtistDataset(Dataset): #default dataset
    def __init__(self, meta, base_path, train=True):
        self.base_path = base_path
        self.meta = meta
        self.target_shape = 81
        classes = meta.artistid.unique()
        self.n_class = len(classes)
        self.aid2class = {c: i for i, c in enumerate(classes)}
        self.train = train
        
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        aid = self.meta.iloc[idx].artistid
        y = self.aid2class[aid]
        
        X = np.load(
            os.path.join(
                self.base_path, 
                self.meta.iloc[idx].archive_features_path
            )
        )
        if X.shape[1] < self.target_shape:
            X = np.pad(X, ((0, 0), (0, self.target_shape - X.shape[1])), mode='reflect')
            
        if self.train:
            if torch.bernoulli(torch.tensor(0.5)).item():
                split = torch.randint(10, 71, (1,)).item()
                X = np.concatenate((X[:, split:], X[:, :split]), axis=1)
            if torch.bernoulli(torch.tensor(0.5)).item():
                mask_start = torch.randint(61, (1,)).item()
                X[:,mask_start:mask_start+20] = 0
                
            if torch.bernoulli(torch.tensor(0.2)).item():
                paths = list(self.meta[self.meta.artistid == aid].archive_features_path)
                chosen = torch.randint(len(paths), (1,)).item()
                X_sup = np.load(os.path.join(self.base_path, paths[chosen]))
                X_sup = np.pad(X_sup, ((0, 0), (0, self.target_shape - X_sup.shape[1])), mode='reflect')
                split = torch.randint(40, (1,)).item()
                X = np.concatenate((X[:, :split+40], X_sup[:, split+40:]), axis=1)
        else:
            X_aug = []
            X_aug.append(X)
            for i in range(3):
                split = torch.randint(10, 71, (1,)).item()
                X_cur = np.concatenate((X[:, split:], X[:, :split]), axis=1)
                X_aug.append(X_cur)
            X = np.array(X_aug)
        
        return X, y
    
    
def round_down(num, divisor):
    return num - (num%divisor)

class ArtistSampler(Sampler): #adopted from voxceleb_trainer by CLOVA
    def __init__(self, dataset, nPerSpeaker, max_seg_per_spk, batch_size, seed=777, **kwargs):
        self.dataset = dataset
        self.nPerSpeaker = nPerSpeaker
        self.max_seg_per_spk = max_seg_per_spk
        self.batch_size = batch_size
        self.epoch = 0
        self.seed = seed
        self.distributed = distributed
        
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)
        indices = torch.randperm(len(self.dataset.meta), generator=g).tolist()
        data_dict = {}
        for index in indices:
            aid = self.dataset.meta.iloc[index].artistid
            speaker_label = self.dataset.aid2class[aid]
            if not (speaker_label in data_dict):
                data_dict[speaker_label] = [];
            data_dict[speaker_label].append(index);
        dictkeys = list(data_dict.keys());
        dictkeys.sort()
        lol = lambda lst, sz: [lst[i:i+sz] for i in range(0, len(lst), sz)]
        flattened_list = []
        flattened_label = []
        for findex, key in enumerate(dictkeys):
            data = data_dict[key]
            numSeg = round_down(min(len(data), self.max_seg_per_spk),self.nPerSpeaker)
            rp = lol(np.arange(numSeg), self.nPerSpeaker)
            flattened_label.extend([findex] * (len(rp)))
            for indices in rp:
                flattened_list.append([data[i] for i in indices])
        mixid = torch.randperm(len(flattened_label), generator=g).tolist()
        mixlabel = []
        mixmap = []
        for ii in mixid:
            startbatch = round_down(len(mixlabel), self.batch_size)
            if flattened_label[ii] not in mixlabel[startbatch:]:
                mixlabel.append(flattened_label[ii])
                mixmap.append(ii)
        mixed_list = [flattened_list[i] for i in mixmap]
        total_size = round_down(len(mixed_list), self.batch_size)
        self.num_samples = total_size
        return iter(mixed_list[:total_size])
        
        
    def __len__(self):
        return self.num_samples
    
class ArtistMultiDataset(Dataset): # add support for many instances per speaker
    def __init__(self, meta, base_path, train=True):
        self.base_path = base_path
        self.meta = meta
        self.target_shape = 81
        classes = meta.artistid.unique()
        self.n_class = len(classes)
        self.aid2class = {c: i for i, c in enumerate(classes)}
        self.train = train
        
    
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, indices):
        if isinstance(indices, int):
            indices = [indices]
        data = []
        label = []
        for idx in indices:
            aid = self.meta.iloc[idx].artistid
            y = self.aid2class[aid]
            label.append(y)
        
            X = np.load(
                os.path.join(
                    self.base_path, 
                    self.meta.iloc[idx].archive_features_path
                )
            )
            if X.shape[1] < self.target_shape:
                X = np.pad(X, ((0, 0), (0, self.target_shape - X.shape[1])), mode='reflect')
                if self.train:
                    if torch.bernoulli(torch.tensor(0.5)).item():
                        split = torch.randint(10, 71, (1,)).item()
                        X = np.concatenate((X[:, split:], X[:, :split]), axis=1)
            data.append(X)
        if len(indices) > 1:
            X = np.stack(data, axis=0)
            y = np.stack(label, axis=0)
        else:
            X = data[0]
            y = label[0]
        return X, y

class CohortDataset(Dataset): #for ASNorm
    def __init__(self, meta, base_path, cohort_idx):
        self.base_path = base_path
        self.meta = meta
        self.cohort_idx = cohort_idx
        self.target_shape = 81
        
    def __len__(self):
        return len(self.cohort_idx)
    
    def __getitem__(self, idx):
        speaker = self.cohort_idx[idx]
        files = self.meta[self.meta.artistid == speaker]
        data = []
        label = []
        for f in files.archive_features_path:
            X = np.load(os.path.join(self.base_path, f))
            if X.shape[1] < self.target_shape:
                X = np.pad(X, ((0,0), (0, self.target_shape - X.shape[1])), mode='reflect')
            data.append(X)
            label.append(speaker)
            
        data = np.array(data)
        label = np.array(label)
        return data, label

class TestDataset(Dataset): #for submission
    def __init__(self, meta, base_path):
        self.base_path = base_path
        self.meta = meta
        self.target_shape = 81
        
    def __len__(self):
        return len(self.meta)
    
    def __getitem__(self, idx):
        tid = int(self.meta.iloc[idx].archive_features_path.split('/')[0])

        X = np.load(
            os.path.join(
                self.base_path, 
                self.meta.iloc[idx].archive_features_path
            )
        )
        if X.shape[1] < self.target_shape:
            X = np.pad(X, ((0, 0), (0, self.target_shape - X.shape[1])), mode='reflect')
        X_aug = []
        X_aug.append(X)
        for i in range(3):
            split = torch.randint(10, 71, (1,)).item()
            X_cur = np.concatenate((X[:, split:], X[:, :split]), axis=1)
            X_aug.append(X_cur)
        X = np.array(X_aug)
            
        return X, tid