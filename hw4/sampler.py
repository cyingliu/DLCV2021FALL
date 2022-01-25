import torch
import numpy as np
from torch.utils.data.sampler import Sampler

class PrototypicalSampler():

    def __init__(self, label, iteration, class_per_it, sample_per_class):
        self.iteration = iteration
        self.class_per_it = class_per_it
        self.sample_per_class = sample_per_class
        self.len = len(label)

        unique_label = []
        for l in label:
            if l not in unique_label:
                unique_label.append(l)
        
        self.indexes_per_class = []
        for ul in unique_label:
            idxs = []
            for idx in range(len(label)):
                if label[idx] == ul: 
                    idxs.append(idx)
            self.indexes_per_class.append(torch.LongTensor(idxs))
        

    def __len__(self):
        return self.iteration
    
    def __iter__(self):
        for i_iter in range(self.iteration):
            batch = []
            classes = torch.randperm(len(self.indexes_per_class))[:self.class_per_it]
            for c in classes:
                l = self.indexes_per_class[c]
                pos = torch.randperm(len(l))[:self.sample_per_class]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

if __name__ == '__main__':

    from dataset import MiniDataset
    from torch.utils.data import DataLoader
    import random

    train_dataset = MiniDataset(csv_path='hw4_data/mini/train.csv', data_dir='hw4_data/mini/train/')
     

    # fix random seeds for reproducibility
    SEED = 123
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(SEED)
    np.random.seed(SEED)

    def worker_init_fn(worker_id):                                                          
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    train_loader = DataLoader(
        train_dataset,
        num_workers=1, pin_memory=True, worker_init_fn=worker_init_fn,
        batch_sampler=PrototypicalSampler(label=train_dataset.data_df['label'].tolist(), 
                                    iteration=1, class_per_it=5, sample_per_class=6))

    for i, (img, label) in enumerate(train_loader):
        print('\ti:', i)
