import torch
import torch.nn as nn
# import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np
from PIL import Image
import pandas as pd
# from torch.utils.tensorboard import SummaryWriter
import sys

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

test_transform = transforms.Compose([                                   
    transforms.ToTensor(),
])

class SatDataset(Dataset):
    def __init__(self, root, transform=None, mode='train'):
        
        self.images = None
        self.labels = None
        self.filenames = []
        self.root = root
        self.transform = transform
        self.mode = mode

        filenames = sorted(glob.glob(os.path.join(root, '*.jpg')))
        for fn in filenames:
            self.filenames.append(fn)

                
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

class fcn32(nn.Module):
  def __init__(self):
    super(fcn32, self).__init__()
    vgg = torchvision.models.vgg16(pretrained=True)
    self.features = vgg.features
    self.classifier = nn.Sequential(
        nn.ConvTranspose2d(512, 7, 32, 32) # [in_channel, out_channel, kernel_size, stride]
    )
  def forward(self, x):
    x = self.features(x) # (batch, 512, 16, 16)
    x = self.classifier(x) # (batch, 7, 512, 512)
    return x

class fcn32_1b(nn.Module): # add one block of convolution
  def __init__(self):
    super(fcn32_1b, self).__init__()
    vgg = torchvision.models.vgg16(pretrained=True)
    self.features = vgg.features
    self.conv = nn.Sequential(
        nn.Conv2d(512, 1024, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 3, 1, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(1024, 1024, 3, 1, 1),
        nn.ReLU(inplace=True)

        )
    self.classifier = nn.Sequential(
        nn.ConvTranspose2d(1024, 7, 32, 32) # [in_channel, out_channel, kernel_size, stride]
    )
  def forward(self, x):
    x = self.features(x) # (batch, 512, 16, 16)
    x = self.conv(x)
    x = self.classifier(x) # (batch, 7, 512, 512)
    return x

class fcn8s(nn.Module):
    # add 1*1 convolution as the paper "Fully Convolutional Networks for Semantic Segmentation" stated
    # fuse predictoin layers
    def __init__(self):
        super(fcn8s, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        self.to_pool3 = nn.Sequential(*list(vgg.features.children())[:17])
        self.to_pool4 = nn.Sequential(*list(vgg.features.children())[17:24])
        self.to_pool5 = nn.Sequential(*list(vgg.features.children())[24:])
        self.to_pool4_predict = nn.Conv2d(512, 7, 1, 1, 0) 
        self.to_pool5_predict = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1), # conv6
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 1024, 3, 1, 1), # conv7
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 7, 1, 1, 0)
            )
        self.pool5_upsample = nn.ConvTranspose2d(7, 7, 2, 2)
        self.to_pool3_predict = nn.Conv2d(256, 7, 1, 1, 0)
        self.fused_pool4_pool5_upsample = nn.ConvTranspose2d(7, 7, 2, 2)
        self.fused_all_upsample = nn.ConvTranspose2d(7, 7, 8, 8)
    def forward(self, x):
        pool3 = self.to_pool3(x)
        pool4 = self.to_pool4(pool3)
        pool5 = self.to_pool5(pool4)
        pool4_predict = self.to_pool4_predict(pool4) # (batch, 7, 32, 32)
        pool5_predict = self.to_pool5_predict(pool5) # (batch, 7, 16, 16)
        pool5_predict_2x = self.pool5_upsample(pool5_predict) # (batch, 7, 32, 32)
        fused_pool4_pool5_predict = pool4_predict + pool5_predict_2x # (batch, 7, 32, 32)
        pool3_predict = self.to_pool3_predict(pool3) # (batch, 7, 64, 64)
        fused_pool4_pool5_predict_2x = self.fused_pool4_pool5_upsample(fused_pool4_pool5_predict) # (batch, 7, 64, 64)
        fused_all = pool3_predict + fused_pool4_pool5_predict_2x
        predict = self.fused_all_upsample(fused_all) # (batch, 7, 512, 512)

        return predict

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

if __name__ == '__main__':

    test_path = sys.argv[1]
    output_dir = sys.argv[2]
    model_path = 'fcn8s-best.pth'

    if not os.path.isdir(output_dir):
        raise ValueError('Invalid output directory')

    model = fcn8s()
    load_checkpoint(model_path, model)
    model.to(device)

    test_dataset = SatDataset(test_path, test_transform, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=1)


    print('predicting...')
    model.eval()
    preds = []
    with torch.no_grad():
      for image in test_dataloader:
        image = image.to(device)
        output = model(image)
        pred = torch.argmax(output, dim=1) # (8, 512, 512)
        preds.append(pred)

    preds = torch.cat(preds, dim=0).cpu().numpy()

    # visualize mask
    print('converting to RGB masks...')
    masks_RGB = np.empty((len(test_dataset), 512, 512, 3))
    for i, p in enumerate(preds):
        masks_RGB[i, p == 0] = [0,255,255]
        masks_RGB[i, p == 1] = [255,255,0]
        masks_RGB[i, p == 2] = [255,0,255]
        masks_RGB[i, p == 3] = [0,255,0]
        masks_RGB[i, p == 4] = [0,0,255]
        masks_RGB[i, p == 5] = [255,255,255]
        masks_RGB[i, p == 6] = [0,0,0]
    masks_RGB = masks_RGB.astype(np.uint8)

    # save image
    print('saving...')
    for mask, filename in zip(masks_RGB, test_dataset.filenames):
        fn = os.path.join(output_dir, filename.replace('jpg', 'png').replace(test_path, '').replace('/', ''))
        im = Image.fromarray(mask)
        im.save(fn)


