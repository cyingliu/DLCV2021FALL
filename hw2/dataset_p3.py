from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import os
import pandas as pd

class DigitDataset(Dataset):
    def __init__(self, root, transform, mode, domain): # mode: 'train', 'test', domain: 'source', 'target', None
        self.transform = transform
        self.fnames =  sorted(glob.glob(os.path.join(root, mode, '*')))
        if domain == 'source':
            df_labels = pd.read_csv(os.path.join(root, "{}.csv".format('train')))
            df_labels.sort_values('image_name')
            self.labels = df_labels['label'].tolist()
        elif mode == 'test':
            df_labels = pd.read_csv(os.path.join(root, "{}.csv".format('test')))
            df_labels.sort_values('image_name')
            self.labels = df_labels['label'].tolist()
        else:
            self.labels = None

        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img

    def __len__(self):
        return self.num_samples

class TestDigitDataset(Dataset):
    def __init__(self, root, transform): # mode: 'train', 'test', domain: 'source', 'target', None
        self.transform = transform
        self.fnames =  sorted(glob.glob(os.path.join(root, '*')))
        self.num_samples = len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        
        return img

    def __len__(self):
        return self.num_samples