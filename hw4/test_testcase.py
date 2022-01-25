import os
import sys
import argparse

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler

import csv
import random
import numpy as np
import pandas as pd

from PIL import Image

from dataset import MiniDataset
from sampler import PrototypicalSampler
from torch.utils.data import DataLoader
from model import Convnet
from utils import euclidean_metric, cosine_similarity_metric, parametric_metric
filenameToPILImage = lambda x: Image.open(x)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def load_checkpoint(checkpoint_path, model, parametric_metric_model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    if parametric_metric_model is not None:
        parametric_metric_model.load_state_dict(state['state_dict_parametric'])
    print('model loaded from %s' % checkpoint_path)

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

# mini-Imagenet dataset
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
        path = self.data_df.loc[index, "filename"]
        label = self.data_df.loc[index, "label"]
        image = self.transform(os.path.join(self.data_dir, path))
        return image, label

    def __len__(self):
        return len(self.data_df)

class GeneratorSampler(Sampler):
    def __init__(self, episode_file_path):
        episode_df = pd.read_csv(episode_file_path).set_index("episode_id")
        self.sampled_sequence = episode_df.values.flatten().tolist()

    def __iter__(self):
        return iter(self.sampled_sequence) 

    def __len__(self):
        return len(self.sampled_sequence)

def predict(args, model, data_loader, parametric_metric_model):
    criterion = nn.CrossEntropyLoss()
    prediction_results = []
    with torch.no_grad():
        # each batch represent one episode (support data + query data)
        for i, (data, target) in enumerate(data_loader):
            data = data.to(device)
            # split data into support and query data
            support_input = data[:args.N_way * args.N_shot,:,:,:] 
            query_input   = data[args.N_way * args.N_shot:,:,:,:]

            # create the relative label (0 ~ N_way-1) for query data
            label_encoder = {target[i * args.N_shot] : i for i in range(args.N_way)}
            query_label = torch.cuda.LongTensor([label_encoder[class_name] for class_name in target[args.N_way * args.N_shot:]]).to(device)

            # TODO: extract the feature of support and query data
            # TODO: calculate the prototype for each class according to its support data
            prototype = model(support_input) 
            prototype = prototype.reshape(args.N_shot, args.N_way, -1).mean(dim=0)

            output = model(query_input)
            if args.dist == 'euclidean':
                logits = euclidean_metric(output, prototype)
            elif args.dist == 'cosine':
                logits = cosine_similarity_metric(output, prototype)
            elif args.dist == 'parametric':
                logits = parametric_metric_model(output, prototype)


            # TODO: classify the query data depending on the its distense with each prototype
            predict = torch.argmax(logits, dim=-1)
            prediction_results.append(predict.cpu().tolist())

    return prediction_results

def parse_args():
    parser = argparse.ArgumentParser(description="Few shot learning")
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--dist', '-d', type=str, default='euclidean') # ['euclidean', 'cosine', 'parametric']
    parser.add_argument('--N-way', default=5, type=int, help='N_way (default: 5)')
    parser.add_argument('--N-shot', default=1, type=int, help='N_shot (default: 1)')
    parser.add_argument('--N-query', default=15, type=int, help='N_query (default: 15)')
    parser.add_argument('--load', type=str, help="Model checkpoint path")
    parser.add_argument('--test_csv', type=str, help="Testing images csv file")
    parser.add_argument('--test_data_dir', type=str, help="Testing images directory")
    parser.add_argument('--testcase_csv', type=str, help="Test case csv")
    parser.add_argument('--output_csv', '-o', type=str, help="Output filename")

    return parser.parse_args()

if __name__=='__main__':
    args = parse_args()
    assert args.dist == 'euclidean' or args.dist == 'cosine' or args.dist == 'parametric'

    test_dataset = MiniDataset(args.test_csv, args.test_data_dir)

    test_loader = DataLoader(
        test_dataset, batch_size=args.N_way * (args.N_query + args.N_shot),
        num_workers=3, pin_memory=False, worker_init_fn=worker_init_fn,
        sampler=GeneratorSampler(args.testcase_csv))

    # TODO: load your model
    checkpoint_path = os.path.join('proto-best.pth')
    model = Convnet()
    if args.dist == 'parametric':
        parametric_metric_model = parametric_metric(args.N_way)
    load_checkpoint(checkpoint_path, model, parametric_metric_model if args.dist == 'parametric' else None)
    if args.dist == 'parametric':
        parametric_metric_model.to(device)
    model.to(device)

    
    prediction_results = predict(args, model, test_loader, parametric_metric_model if args.dist == 'parametric' else None)

    # TODO: output your prediction to csv
    fout = open(args.output_csv, 'w')
    header_line = 'episode_id,' + ','.join(['query{}'.format(i) for i in range(75)])
    fout.write(header_line + '\n')
    for i in range(len(prediction_results)):
        line = '{},'.format(str(i)) + ','.join([str(p) for p in prediction_results[i]])
        fout.write(line + '\n')
    fout.close()
