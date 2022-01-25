from digit_classifier import Classifier 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch
import glob
from PIL import Image
import os
import argparse

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

class DigitDataset(Dataset):
    def __init__(self, root, transform):
        self.transform = transform
        self.fnames =  sorted(glob.glob(os.path.join(root, '*')))
        self.labels = []
        for fn in self.fnames:
            self.labels.append(int(fn.replace(root, '').replace('/', '').split('_')[0]))
    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = Image.open(fname)
        img = self.transform(img)
        label = self.labels[idx]
        return img, label
    def __len__(self):
        return len(self.fnames)

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--epoch', '-e', type=str)
    # parser.add_argument('--exp_name', '-n', type=str)
    # args = parser.parse_args()
    # image_dir = os.path.join('p2_sample', args.exp_name)
    image_dir = 'p2_output/'
    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5] ) ] )
    test_dataset = DigitDataset(image_dir, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)
    model = Classifier()
    model.to(device)
    load_checkpoint('Classifier_oldformat.pth', model)
    correct = 0
    model.eval()
    for image, label in test_dataloader:
        image = image.to(device)
        with torch.no_grad():
            logit = model(image)
            predict = torch.argmax(logit, dim=-1).cpu()
        correct += torch.sum(predict == label.view(predict.shape)).item()
    print('Avg acc: {}'.format(correct / len(test_dataset)))

