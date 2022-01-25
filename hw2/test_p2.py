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
# import argparse
import torchvision.transforms as transforms
from model_p2 import Generator_fmnist, Generator, Generator_64
from dataset_p2 import MNISTMDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model_G):
    state_dict = torch.load(checkpoint_path)
    model_G.load_state_dict(state_dict['state_dict_G'])
    print('model loaded from %s' % checkpoint_path)

def label2onehot(label):
    # print(label.shape, label)
    label_onehot = torch.zeros(label.shape[0], 10).to(device)
    label_onehot = label_onehot.scatter_(1, label, 1).view(label.shape[0], 10)
    return label_onehot

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', '-e', type=str)
    # parser.add_argument('--exp_name', '-n', type=str)
    # parser.add_argument('--G_emb_dim', type=int, default=50)
    # parser.add_argument('--G_latent_dim', type=int, default=384)
    # parser.add_argument('--G_n_channel', type=int, default=192)
    # parser.add_argument('--z_dim', type=int, default=100)
    # args = parser.parse_args()
    # print(args)
    model_path = 'acgan.pth'
    save_image_dir = sys.argv[1]
    z_dim = 100

    model_G = Generator_64() # emb_dim=args.G_emb_dim, latent_dim=args.G_latent_dim, n_channel=args.G_n_channel
    load_checkpoint(model_path, model_G)
    model_G.to(device)
    z_sample = torch.randn(1000, z_dim).to(device)
    sample_label = torch.cat([torch.arange(10) for i in range(100)], dim=0).long().unsqueeze(-1).to(device)
    sample_label_onehot = label2onehot(sample_label).long()
    model_G.eval()
    fake_imgs_sample = (model_G(z_sample, sample_label_onehot).data + 1) / 2.0
    
    transform = transforms.Compose(
        [transforms.ToPILImage(),
        transforms.Resize((28, 28)),
        transforms.ToTensor()])
    
    resized_samples = []
    
    for img in fake_imgs_sample:
        resized_samples.append(transform(img.cpu()))
    # torchvision.utils.save_image(resized_samples[:100], 'p2_sample.jpg', nrow=10)
    cnt = 0
    for n in range(100):
        for i in range(10):
            filename = os.path.join(save_image_dir, '{}_{}.png'.format(i, str(n).zfill(3)))
            torchvision.utils.save_image(resized_samples[cnt], filename, nrow=1)
            cnt += 1

