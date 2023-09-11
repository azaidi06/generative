from torch.utils.data import Dataset, DataLoader
from fastai.vision.all import *
import numpy as np
import pandas as pd
import PIL
import torch


def get_mnist_df(train=True):
    if train is True: 
        path = Path('data/mnist_png/training/')
    else: path = Path('data/mnist_png/testing/')
    
    data = {idx:parent_dir.ls() for idx, parent_dir in enumerate(path.ls())}
    df = [pd.DataFrame(list(data[e]), columns=['path']) for e in data]
    df = pd.concat(df)
    df['label'] = df['path'].map(lambda x: str(x).split('/')[-2])
    return df


def get_img(path):
    return Image.open(path)

def to_npy(x): return np.array(x)


class ImageDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = to_npy(Image.open(row.path))
        img_tns = torch.tensor(img).unsqueeze(0).float()
        lbl = row.label
        #img_tns = torch.nn.functional.pad(img_tns, (2,2,2,2))
        return img_tns, img_tns
    
    
class LabeledDataset(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = to_npy(Image.open(row.path))
        img_tns = torch.tensor(img).unsqueeze(0).float()
        lbl = row.label
        return img_tns, lbl
    

def get_dl(train=False, bs=256, shuffle=True, num_workers=8):
    df = get_mnist_df(train=train)
    ds = LabeledDataset(df)
    return DataLoader(ds, batch_size=512, shuffle=False)


def get_dls(train_df=None, valid_df=None, bs=64):
    if train_df is None:
        train_df = get_mnist_df()
        valid_df = get_mnist_df(train=False)
    train_ds = ImageDataset(train_df)
    valid_ds = ImageDataset(valid_df)
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8)
    valid_dl = DataLoader(valid_ds, batch_size=bs*2, shuffle=False, num_workers=8)
    return train_dl, valid_dl