import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset

import csv
import numpy as np
import pandas as pd

from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

class MiniDataset(Dataset):
    def __init__(self, csv_path, data_dir):
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")

        self.transform = transforms.Compose([
            filenameToPILImage,
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        index = index.item()
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':

    train_dataset = MiniDataset(csv_path='hw4_data/mini/train.csv', data_dir='hw4_data/mini/train/')
    print(train_dataset[0][0].shape)
    print(train_dataset[0][1])
    print(train_dataset.data_df['label'].tolist())