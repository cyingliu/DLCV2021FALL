import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import sys
import numpy as np
from PIL import Image
import pandas as pd
import random

from pytorch_pretrained_vit import ViT
from dataset import ImageDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(100)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    input_dir, output_path = sys.argv[1], sys.argv[2]

    checkpoint_path = 'vit_best.pth'

    

    model = ViT('B_16_imagenet1k')
    model.fc = nn.Linear(768, 37)
    load_checkpoint(checkpoint_path, model)
    model.eval()
    model.to(device)

    transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    test_dataset = ImageDataset(root=input_dir, transform=transform, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2)

    # correct = 0
    predicts = []
    with torch.no_grad():
        for i, (imgs) in enumerate(test_dataloader):
            imgs = imgs.to(device)
            logits = model(imgs)
            predict = torch.argmax(logits, dim=-1)
            # correct += torch.sum(predict == labels.view(predict.shape)).item()
            predicts.extend(predict.cpu().tolist())

    # write to output file
    fout = open(output_path, 'w')
    fout.write('filename,label\n')
    for i in range(len(predicts)):
        fname = test_dataset.fnames[i].replace(input_dir, '').replace('/', '')
        fout.write('{},{}\n'.format(fname, predicts[i]))
    fout.close()
