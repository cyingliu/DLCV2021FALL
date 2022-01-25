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
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import random
import argparse

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

train_transform = transforms.Compose([                                   
    transforms.ToTensor(),
])
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

        filenames = sorted(glob.glob(os.path.join(root, '*_sat.jpg')))
        for fn in filenames:
            mask_fn = fn.replace('sat', 'mask').replace('jpg', 'png')
            self.filenames.append((fn, mask_fn)) # (img filename, mask filename) pair
                
        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image_fn, mask_fn = self.filenames[index]
        image = Image.open(image_fn)
        
        masks = np.empty((512, 512))
        mask = Image.open(mask_fn)

        if self.mode == 'train':
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
        mask = np.array(mask)

        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        masks[mask == 3] = 0  # (Cyan: 011) Urban land 
        masks[mask == 6] = 1  # (Yellow: 110) Agriculture land 
        masks[mask == 5] = 2  # (Purple: 101) Rangeland 
        masks[mask == 2] = 3  # (Green: 010) Forest land 
        masks[mask == 1] = 4  # (Blue: 001) Water 
        masks[mask == 7] = 5  # (White: 111) Barren land 
        masks[mask == 0] = 6  # (Black: 000) Unknown
        # deal with invalid masks
        masks[masks > 6] = 0
        masks[masks < 0] = 0
        masks = torch.LongTensor(masks)
        
        if self.transform is not None:
            image = self.transform(image)


        
        return image, masks

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

class fcn8(nn.Module):
    def __init__(self):
        super(fcn8, self).__init__()
        vgg = torchvision.models.vgg16(pretrained=True)
        # out_channel, pool3: 256, pool4: 512, pool5: 512
        self.to_pool3 = nn.Sequential(*list(vgg.features.children())[:17])
        self.to_pool4 = nn.Sequential(*list(vgg.features.children())[17:24])
        self.to_pool5 = nn.Sequential(*list(vgg.features.children())[24:])
        self.conv3 = nn.Conv2d(256, 512, 3, 1, 1)
        self.upsample2 = nn.ConvTranspose2d(512, 512, 2, 2)
        self.upsample4 = nn.ConvTranspose2d(512, 512, 4, 4)
        self.upsample8 = nn.ConvTranspose2d(512, 7, 8, 8)

    def forward(self, x):
        pool3 = self.to_pool3(x)
        pool4 = self.to_pool4(pool3)
        pool5 = self.to_pool5(pool4)
        pool3_2ch = self.conv3(pool3)
        pool4_2x = self.upsample2(pool4)
        pool5_4x = self.upsample4(pool5)
        out = self.upsample8(pool3_2ch + pool4_2x + pool5_4x)
        return out

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

def save_checkpoint(checkpoint_path, model):
    state = {'state_dict': model.state_dict()}
    torch.save(state, checkpoint_path) ## save as old format
    print('model saved to %s' % checkpoint_path)
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
        iou = tp / (tp_fp + tp_fn - tp + 1e-10)
        mean_iou += iou / 6
        print('class #%d : %1.5f'%(i, iou))
    print('\nmean_iou: %f\n' % mean_iou)

    return mean_iou

def booltostr(x):
    if x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        raise ValueError('Invalid scheduler argument')

if __name__ == '__main__':

    # set up
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', '-t_pth', type=str, default='p2_data/train/')
    parser.add_argument('--valid_path', '-v_pth', type=str, default='p2_data/validation/')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--lr', '-lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', '-bs', type=int, default=4)
    parser.add_argument('--epoch', '-e', type=int, default=75)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=5000)
    parser.add_argument('--optimizer', '-o', type=str, default='Adam', help='Adam or SGD')
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--scheduler', type=booltostr, default=False)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--warm_up_steps', type=int, default=8000)
    parser.add_argument('--scheduler_step', type=int, default=100)
    args = parser.parse_args()
    print(args)

    checkpoint_dir = os.path.join('./result', args.exp_name)
    log_dir = os.path.join(checkpoint_dir, 'log')
    result_path = os.path.join(checkpoint_dir, '{}.csv'.format(args.exp_name))
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    train_dataset = SatDataset(args.train_path, train_transform, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    val_dataset = SatDataset(args.valid_path, test_transform, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

    model = fcn8s() ## model
    model.to(device)
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
        raise ValueError('Invalid optimizer')
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([2.,1.,2.,2.,2.,2.,2.]))

    if args.scheduler == True:
        warm_up_step_lr = lambda step: (step + 1) / (args.warm_up_steps) if step < args.warm_up_steps else args.gamma ** ((step - args.warm_up_steps) // args.scheduler_step)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, warm_up_step_lr)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

    best_IOU = 0
    iteration = 0
    total_iter = args.epoch * len(train_dataloader)
    temp_loss = 0

    for e in range(args.epoch):
        model.train()
        for image, mask in train_dataloader:
          image, mask = image.to(device), mask.to(device)
          output = model(image)
          loss = criterion(output, mask)
          temp_loss += loss
          if iteration % args.grad_step == 0:
            temp_loss.backward()
            optimizer.step()
            if args.scheduler: scheduler.step()
            optimizer.zero_grad()
            temp_loss = 0
          if iteration % args.log_interval == 0:
            pred = torch.argmax(output, dim=1)
            iou_score = mean_iou_score(pred.cpu().numpy(), mask.cpu().numpy())
            train_writer.add_scalar('loss', loss.item(), iteration)
            train_writer.add_scalar('iou', iou_score, iteration)
            train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
            print('Epoch: {}\t[{}/{}]\tLoss: {}\tIou: {}\tlr: {}'.format(e, iteration, total_iter, loss.item(), iou_score, optimizer.param_groups[0]['lr']))
          if iteration % args.save_interval == 0:
            save_checkpoint(os.path.join(checkpoint_dir, '{}-{}.pth'.format(args.exp_name, iteration)), model)
          iteration += 1

        # validation
        model.eval()
        preds = []
        targets = []
        temp_loss = 0
        with torch.no_grad():
          for image, mask in val_dataloader:
            image, mask = image.to(device), mask.to(device)
            output = model(image)
            loss = criterion(output, mask)
            pred = torch.argmax(output, dim=1) # (8, 512, 512)
            targets.append(mask)
            preds.append(pred)
            temp_loss += loss.item()
        preds = torch.cat(preds, dim=0).cpu().numpy()
        targets = torch.cat(targets, dim=0).cpu().numpy()
        iou_score = mean_iou_score(preds, targets)
        val_writer.add_scalar('loss', temp_loss/len(val_dataloader), iteration)
        val_writer.add_scalar('iou', iou_score, iteration)
        print('[VAL] Epoch: {}\t[{}/{}]\tLoss: {}\tIou: {}'.format(e, iteration, total_iter, temp_loss/len(val_dataloader), iou_score))
        if iou_score >= best_IOU:
          best_IOU = iou_score
          save_checkpoint(os.path.join(checkpoint_dir, '{}-best.pth'.format(args.exp_name)), model)
          print('BEST model saved')

    save_checkpoint(os.path.join(checkpoint_dir, '{}-{}.pth'.format(args.exp_name, iteration)), model)





