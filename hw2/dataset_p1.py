from torch.utils.data import Dataset
import torchvision.transforms as transforms
import glob
from PIL import Image
import os

class FaceDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.fnames =  glob.glob(os.path.join(root, '*'))
        self.num_samples = len(self.fnames)
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples



if __name__ == '__main__':
    
    from torch.utils.data import DataLoader
    import torchvision
    
    data_dir = 'hw2_data/face/train/'
    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    train_dataset = FaceDataset(data_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False, num_workers=2)
    generator = iter(train_dataloader)
    data = generator.next() # (batch, 3, 64, 64)
    print(data.shape)

    #============= save imgs ================#
    imgs_sample = data + 1 / 2.0
    filename = 'sample.jpg'
    torchvision.utils.save_image(imgs_sample, filename, nrow=2)
