import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import numpy as np
from PIL import Image
import pandas as pd
import sys
import os

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_transform = transforms.Compose([
    transforms.Resize(size=(40,40)),                                    
    transforms.ToTensor(),
])
class ImageDataset(Dataset):
    def __init__(self, root, transform=None):
        """ Intialize the MNIST dataset """
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform

        # read filenames
        filenames = sorted(glob.glob(os.path.join(root, '*.png')))
        for fn in filenames:
            self.filenames.append(fn) # (filename) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class VGG16_custom(nn.Module):
  def __init__(self):
    super(VGG16_custom, self).__init__()
    vgg16 = torchvision.models.vgg16(pretrained=False)
    self.backbone = vgg16.features
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    self.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 50) 
    )
  def forward(self, x):
    x = self.backbone(x)
    x = self.avgpool(x)
    x = x.view(-1, 512)
    x = self.fc(x)
    return x

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    test_dir, output_path =  sys.argv[1], sys.argv[2]
    test_dataset = ImageDataset(test_dir, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=1)

    # load models
    vgg16 = VGG16_custom()
    load_checkpoint('vgg16-best.pth', vgg16)

    resnet152 = torchvision.models.resnet152(pretrained=False)
    resnet152.fc = nn.Linear(2048, 50)
    load_checkpoint('resnet152-best.pth', resnet152)

    dense161 = torchvision.models.densenet161(pretrained=False)
    dense161.classifier = nn.Linear(2208, 50)
    load_checkpoint('dense161-best.pth', dense161)

    dense169 = torchvision.models.densenet169(pretrained=False)
    dense169.classifier = nn.Linear(1664, 50)
    load_checkpoint('dense169-best.pth', dense169)

    resnext = torchvision.models.resnext101_32x8d(pretrained=False)
    resnext.fc = nn.Linear(2048, 50)
    load_checkpoint('resnext-best.pth', resnext)


    vgg19_bn1 = torchvision.models.vgg19_bn(pretrained=False)
    vgg19_bn1.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 50)
    )
    load_checkpoint('vgg19_bn-best.pth', vgg19_bn1)

    vgg19_bn2 = torchvision.models.vgg19_bn(pretrained=False)
    vgg19_bn2.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 50)
    )
    load_checkpoint('vgg19_bn2-best.pth', vgg19_bn2)

    vgg19_bn3 = torchvision.models.vgg19_bn(pretrained=False)
    vgg19_bn3.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(4096, 2048),
        nn.ReLU(),
        nn.Dropout(),
        nn.Linear(2048, 50)
    )
    load_checkpoint('vgg19_bn3-best.pth', vgg19_bn3)

    models_dict = {'vgg16': vgg16, 'resnet152': resnet152, 'dense161': dense161, 'dense169': dense169, 'resnext': resnext,
            'vgg19_bn1': vgg19_bn1, 'vgg19_bn2': vgg19_bn2, 'vgg19_bn3': vgg19_bn3}

    models_weight = pd.read_csv('en_weight.csv')

    preds = []
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            y_hats_temp = dict()
            for model_name, model in models_dict.items():
              model.eval()
              model.to(device)
              output = model(data)
              y_hats_temp[model_name] = output
              model.cpu()
            output = sum([y_hats_temp[model_name] * models_weight.loc[models_weight['model']== model_name, 'weight'].values[0] for model_name in models_dict.keys()])
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            preds.extend(pred.squeeze(-1).tolist())

    # write to csv
    fout = open(output_path, 'w')
    fout.write('image_id,label\n')
    for i in range(len(test_dataset)):
      fout.write('{},{}\n'.format(test_dataset.filenames[i].replace(test_dir, '').replace('/', ''), preds[i]))
    fout.close()    