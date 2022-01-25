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
import matplotlib.pyplot as plt
import pandas as pd

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

#training 時做 data augmentation
train_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=(40, 40)),
    # transforms.RandomCrop(size=(32, 32)),
    transforms.RandomHorizontalFlip(), #隨機將圖片水平翻轉
    transforms.RandomRotation(15), #隨機旋轉圖片
    transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)), # 參考同學分享
    transforms.RandomPerspective(distortion_scale=0.2,p=0.5), # 參考同學分享
    # transforms.ColorJitter(brightness=0.5),
    # transforms.RandomResizedCrop(size = 128, scale=(0.1, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
    transforms.ToTensor(), #將圖片轉成 Tensor，並把數值normalize到[0,1](data normalization)
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to (-1, 1)
])
#testing 時不需做 data augmentation
test_transform = transforms.Compose([
    # transforms.ToPILImage(),
    transforms.Resize(size=(40,40)),                                    
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # normalize to (-1, 1)
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
        for i in range(50):
            filenames = sorted(glob.glob(os.path.join(root, '{}_*.png'.format(i))))
            for fn in filenames:
                self.filenames.append((fn, i)) # (filename, label) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, label = self.filenames[index]
        image = Image.open(image_fn)
            
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

class VGG16_custom(nn.Module):
  def __init__(self):
    super(VGG16_custom, self).__init__()
    vgg16 = torchvision.models.vgg16(pretrained=True)
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

def save_checkpoint(checkpoint_path, model): # optimizer
    state = {'state_dict': model.state_dict()}
            #  'optimizer' : optimizer.state_dict()}
    torch.save(state, checkpoint_path, _use_new_zipfile_serialization=False) ## save as old format
    print('model saved to %s' % checkpoint_path)
def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    # optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def write_logs(train_losses, val_losses, val_accs, val_iters):
  train_loss_path = os.path.join(log_dir, 'train_loss.txt')
  val_loss_path = os.path.join(log_dir, 'val_loss.txt')
  val_acc_path = os.path.join(log_dir, 'val_acc.txt')
  val_iter_path = os.path.join(log_dir, 'val_iter.txt')
  # lr_path = os.path.join(log_dir, 'lr.txt')
  fout = open(train_loss_path, 'w')
  for x in train_losses:
    fout.write('{}\n'.format(x))
  fout.close()
  fout = open(val_loss_path, 'w')
  for x in val_losses:
    fout.write('{}\n'.format(x))
  fout.close()
  fout = open(val_acc_path, 'w')
  for x in val_accs:
    fout.write('{}\n'.format(x))
  fout.close()
  fout = open(val_iter_path, 'w')
  for x in val_iters:
    fout.write('{}\n'.format(x))
  fout.close()

def test(model):
    criterion = nn.CrossEntropyLoss()
    model.eval()  # Important: set evaluation mode
    test_loss = 0
    correct = 0
    with torch.no_grad(): # This will free the GPU memory used for back-prop
        for data, target in val_dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_dataloader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(val_dataloader.dataset),
        100. * correct / len(val_dataloader.dataset)))
    return test_loss, correct

def train(model, epoch, save_interval=20000, log_interval=100):

    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8000, gamma=0.7)

    criterion = nn.CrossEntropyLoss()
    model.train()  # Important: set training mode
    
    train_losses = []
    val_losses = []
    val_accs = []
    val_iters = []
    lrs = []
    
    best_correct = 0
    iteration = 42240
    for ep in range(epoch):
        for batch_idx, (data, target) in enumerate(train_dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if iteration % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tlr: {}'.format(
                    ep, batch_idx * len(data), len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), loss.item(), scheduler.get_last_lr()[0]))
            if iteration % save_interval == 0:
              # save_checkpoint(os.path.join(checkpoint_dir, '{}-{}.pth'.format(exp_name, iteration)), model, optimizer)
              write_logs(train_losses, val_losses, val_accs, val_iters)
              print('log saved')
            train_losses.append(loss.item())
            lrs.append(scheduler.get_last_lr()[0])
            iteration += 1

        loss, correct = test(model) # Evaluate at the end of each epoch
        model.train()
        val_losses.append(loss)
        val_accs.append(correct / len(val_dataloader.dataset))
        val_iters.append(iteration)
        if correct >= best_correct:
          best_correct = correct
          save_checkpoint(os.path.join(checkpoint_dir, '{}-best.pth'.format(exp_name)), model, optimizer)
          print('best model saved')

    save_checkpoint(os.path.join(checkpoint_dir, '{}-{}.pth'.format(exp_name, iteration)), model, optimizer)
    write_logs(train_losses, val_losses, val_accs, val_iters)
    print('model saved')

if __name__ == '__main__':
    train_path = 'p1_data/train_50/'
    valid_path = 'p1_data/val_50/'
    exp_name = 'tmp'
    model_type = 'vgg16'
    checkpoint_dir = os.path.join('./result_p1', exp_name)
    log_dir = os.path.join(checkpoint_dir, 'log')
    result_path = os.path.join(checkpoint_dir, '{}.csv'.format(exp_name))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    train_dataset = ImageDataset(train_path, transform=train_transform) # transforms.ToTensor()
    val_dataset = ImageDataset(valid_path, transform=test_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=1)

    if model_type == 'vgg16':
        model = VGG16_custom()
    elif model_type == 'resnet152':
        model = torchvision.models.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, 50)
    elif model_type == 'dense161':
        model = torchvision.models.densenet161(pretrained=True)
        model.classifier = nn.Linear(2208, 50)
    elif model_type == 'resnext':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
        model.fc = nn.Linear(2048, 50)
    elif model_type == 'vgg19_bn1':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 50)
        )
    elif model_type == 'vgg19_bn2':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 50)
        )
    elif model_type == 'vgg19_bn3':
        model = torchvision.models.vgg19_bn(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(2048, 50)
        )
    else:
        raise ValueError('Invalid model type')

    model.to(device)
    train(model, 120)

