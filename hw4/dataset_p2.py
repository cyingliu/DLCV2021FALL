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
            transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    def __getitem__(self, index):
        # index = index.item()
        path = self.data_df.loc[index, "filename"]
        # label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image

    def __len__(self):
        return len(self.data_df)

class OfficeDataset(Dataset):
    def __init__(self, csv_path, data_dir, label2id_path, transform, mode='train'): # mode: 'train', 'val', 'test'
        self.data_dir = data_dir
        self.data_df = pd.read_csv(csv_path).set_index("id")
       	label2id_df = pd.read_csv(label2id_path)
       	self.label2id = {k: v for (k, v) in zip(label2id_df['label'].tolist(), label2id_df['id'].tolist())}

        self.transform = transform
        self.mode = mode

    def __getitem__(self, index):
        path = self.data_df.loc[index, "filename"]
        image = self.transform(os.path.join(self.data_dir, path))
        if self.mode == 'train' or self.mode == 'val':
	        label = self.label2id[self.data_df.loc[index, "label"]]
        	return image, label
        else:
        	return image

    def __len__(self):
        return len(self.data_df)

if __name__ == '__main__':

	from torch.utils.data import DataLoader
	train_dataset = OfficeDataset(csv_path='hw4_data/office/val.csv', data_dir='hw4_data/office/val', label2id_path='label2id.csv', mode='val')
	train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=False)
	generator = iter(train_dataloader)
	image, label = generator.next()
	print(image.shape, label)

	# fin = open('hw4_data/office/val.csv', 'r')
	# lines = fin.readlines()
	# fin.close()
	# label2id = {}
	# cnt = 0
	# for line in lines[1:]:
	# 	_id, filename, label = line.strip().split(',')
	# 	if label not in label2id:
	# 		label2id[label] = cnt
	# 		cnt += 1
	# fout = open('label2id.csv', 'w')
	# fout.write('label,id\n')
	# for k, v in label2id.items():
	# 	fout.write('{},{}\n'.format(k,v))
	# fout.close()
