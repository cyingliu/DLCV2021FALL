from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import os
import pandas as pd

class MNISTMDataset(Dataset):
    def __init__(self, root, transform, mode='train'): # mode: 'train' or 'test'
        self.transform = transform
        self.fnames =  sorted(glob.glob(os.path.join(root, mode, '*')))
        df_labels = pd.read_csv(os.path.join(root, "{}.csv".format(mode)))
        df_labels.sort_values('image_name')
        self.labels = df_labels['label'].tolist()


        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        label = self.labels[idx]
        # print(fname, label)
        return img, label

    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    import torchvision
    
    data_dir = 'hw2_data/digits/mnistm'
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ) ] )
    train_dataset = MNISTMDataset(data_dir, transform, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=False, num_workers=1)
    generator = iter(train_dataloader)
    img, label = generator.next() # (batch, 3, 64, 64)
    print(img.shape)

    #============= save imgs ================#
    print('label:\n', label)
    imgs_sample = (img + 1) / 2.0
    filename = 'sample_digit.jpg'
    torchvision.utils.save_image(imgs_sample, filename, nrow=2)
