import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import pandas as pd
import random
# import argparse
import sys

from model_p1 import Discriminator, Generator
from dataset_p1 import FaceDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict_G'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', type=str, help='exp name')
    # parser.add_argument('-e', type=int, help='num of epoch')
    # parser.add_argument('-s', type=int, help='random seed', default=123)
    # args = parser.parse_args()
    torch.manual_seed(123)
    z_dim = 100
    save_dir = sys.argv[1]
    checkpoint_path = 'dcgan.pth'
    model_G = Generator(in_dim=z_dim)
    # model_G.load_state_dict(torch.load(checkpoint_path))
    load_checkpoint(checkpoint_path, model_G)
    model_G.to(device)
    z_sample = torch.randn(1000, z_dim, 1, 1).to(device)
    model_G.eval()
    with torch.no_grad():
        fake_imgs_sample = (model_G(z_sample).data + 1) / 2.0
    # torchvision.utils.save_image(fake_imgs_sample[:32], 'p1_sample.jpg', nrow=8)
    for i in range(fake_imgs_sample.shape[0]):
        filename = os.path.join(save_dir, '{}.png'.format(str(i).zfill(4)))
        torchvision.utils.save_image(fake_imgs_sample[i], filename, nrow=1)

