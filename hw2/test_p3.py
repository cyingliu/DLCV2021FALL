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

from dataset_p3 import TestDigitDataset
from model_p3 import DANN_svhn1, DANN_svhn2, DANN_svhn3, DANN_svhn4, DANN_small

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def test_svhn(data_dir, output_file):
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])
    test_dataset = TestDigitDataset(data_dir, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    model1 = DANN_svhn1()
    model2 = DANN_svhn2()
    model3 = DANN_svhn3()
    model4 = DANN_svhn4()

    load_checkpoint('dann_svhn1.pth', model1)
    load_checkpoint('dann_svhn2.pth', model2)
    load_checkpoint('dann_svhn3.pth', model3)
    load_checkpoint('dann_svhn4.pth', model4)

    models_dict = {'model1': model1, 'model2': model2, 'model3': model3, 'model4': model4}

    # ensemble by voting
    preds = []
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            preds_collect = []
            pred = []
            for model_name, model in models_dict.items():
                model.eval()
                model.to(device)
                output, _ = model(data, alpha=-1)
                pred_temp = output.max(1, keepdim=True)[1].cpu().numpy()
                preds_collect.append(pred_temp)
                model.cpu()
            preds_collect = np.array(preds_collect).squeeze(-1) # (num of model, batch)
            for i in range(preds_collect.shape[1]):
                counts = np.bincount(preds_collect[:, i], minlength=10)
                pred.append(np.argmax(counts))
            pred = np.array(pred)
            preds.extend(pred.tolist())

    fout = open(output_file, 'w')
    fout.write('image_name,label\n')
    for i in range(len(preds)):
        fname = test_dataset.fnames[i].split('/')[-1]
        fout.write('{},{}\n'.format(fname, preds[i]))
    fout.close()

def test_usps(data_dir, output_file):
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # for usps
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    test_dataset = TestDigitDataset(data_dir, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    model = DANN_small()
    load_checkpoint('dann_usps.pth', model)
    model.to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            output, _ = model(data, alpha=-1)
            pred = output.max(1, keepdim=True)[1].cpu().numpy()
            preds.extend(pred.reshape(-1).tolist())

    fout = open(output_file, 'w')
    fout.write('image_name,label\n')
    for i in range(len(preds)):
        fname = test_dataset.fnames[i].split('/')[-1]
        fout.write('{},{}\n'.format(fname, preds[i]))
    fout.close()

def test_mnistm(data_dir, output_file):
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    test_dataset = TestDigitDataset(data_dir, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    model = DANN_small()
    load_checkpoint('dann_mnistm.pth', model)
    model.to(device)

    preds = []
    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            output, _ = model(data, alpha=-1)
            pred = output.max(1, keepdim=True)[1].cpu().numpy()
            preds.extend(pred.reshape(-1).tolist())

    fout = open(output_file, 'w')
    fout.write('image_name,label\n')
    for i in range(len(preds)):
        fname = test_dataset.fnames[i].split('/')[-1]
        fout.write('{},{}\n'.format(fname, preds[i]))
    fout.close()


if __name__ == '__main__':

    data_dir = sys.argv[1]
    domain_name = sys.argv[2]
    output_file = sys.argv[3]

    if domain_name == 'mnistm':
        test_mnistm(data_dir, output_file)
    elif domain_name == 'usps':
        test_usps(data_dir, output_file)
    elif domain_name == 'svhn':
        test_svhn(data_dir, output_file)
