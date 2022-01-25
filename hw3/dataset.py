from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import os
import pandas as pd
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transform, mode): # mode: 'train', 'val', 'test'
        self.transform = transform
        self.fnames =  sorted(glob.glob(os.path.join(root, '*')))
        if mode == 'train':
            self.fnames.remove(os.path.join(root, '11_2395.jpg'))
            self.fnames.remove(os.path.join(root, '11_2427.jpg'))
        self.labels = []
        if mode == 'train' or mode == 'val':
            for f in self.fnames:
                img_name = f.replace(root, '').replace('/', '')
                self.labels.append(int(img_name.split('_')[0]))

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        
        if np.array(img).shape[2] > 3:
            img_arr = np.array(img)[:, :, :3]
            img = Image.fromarray(img_arr)
        
        img = self.transform(img)
        if len(self.labels) > 0:
            label = self.labels[idx]
            return img, label
        else:
            return img

    def __len__(self):
        return len(self.fnames)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import torch

    # Use GPU if available, otherwise stick with cpu
    use_cuda = torch.cuda.is_available()
    torch.manual_seed(123)
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)

    data_dir = 'hw3_data/p1_data/train/'
    transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ] )

    train_dataset = ImageDataset(root=data_dir, transform=transform, mode='val')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    generator = iter(train_dataloader)
    img, class_label = generator.next()
    img, class_label = img.to(device), class_label.to(device)

    for img, label in train_dataloader:
        pass
