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
import argparse
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt

from dataset_p3 import DigitDataset
from model_p3 import DANN_svhn1, DANN_small

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_data_dir', '-ds', type=str, default='hw2_data/digits/mnistm/')
    parser.add_argument('--target_data_dir', '-dt', type=str, default='hw2_data/digits/usps/')
    parser.add_argument('--output_path', '-op', type=str, default='p3_vis/a_1.png')
    parser.add_argument('--model_path', '-mp', type=str, default='dann_usps.pth')
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--title', '-t', type=str, default='mnistm to usps')
    args = parser.parse_args()
    print(args)

    size = 28 if 'usps' in args.source_data_dir else 64
    
    transform = transforms.Compose(
        [transforms.Resize((size, size)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    transform_usps = transforms.Compose(
        [transforms.Resize((size, size)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # for usps
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

    source_dataset = DigitDataset(args.source_data_dir, 
                                    transform if 'usps' not in args.source_data_dir else transform_usps, 
                                    mode='test', domain=None)
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    target_dataset = DigitDataset(args.target_data_dir, 
                                    transform if 'usps' not in args.target_data_dir else transform_usps, 
                                    mode='test', domain=None)
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    if 'usps' in args.source_data_dir:
        model = DANN_svhn1()
    else:
        model = DANN_small()

    load_checkpoint(args.model_path, model)
    model.to(device)
    model.eval()

    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    model.feature_extractor[-1].register_forward_hook(get_activation('feature_extractor[-1]'))

    indice = []
    cnt = 0
    for dataloader in [source_dataloader, target_dataloader]:
        domain_name = 'source' if cnt == 0 else 'target'
        cnt += 1

        print('computing activation map...')
        inter_outputs = []
        targets = []
        with torch.no_grad():
          for i, (data, target) in enumerate(source_dataloader):
            if i > 30: break
            data = data.cuda()
            output = model(data, alpha=-1)
            inter_output = activation['feature_extractor[-1]']
            inter_output = inter_output.view(data.shape[0], -1).detach().cpu().numpy().astype('float64')
            inter_outputs.append(inter_output)
            targets.append(target.view(-1, 1).numpy())
        inter_outputs = np.vstack(inter_outputs)
        targets = np.vstack(targets).squeeze(-1)


        print('computing TSNE...')
        X_embedded = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(inter_outputs)

        print('plotting...')
        rgb_values = sns.color_palette("husl", 2)
        color_map = dict(zip(['source', 'target'], rgb_values))
        for vec, label in zip(X_embedded, targets):
          plt.scatter(vec[0], vec[1], color=color_map[domain_name], label=domain_name if domain_name not in indice else '', s=2)
          if domain_name not in indice:
            indice.append(domain_name)
    
    plt.legend(loc='lower right')
    plt.title(args.title)
    plt.savefig(args.output_path)
