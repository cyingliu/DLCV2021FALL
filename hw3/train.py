import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
import glob
import os
import numpy as np
from PIL import Image
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import random
import argparse

from pytorch_pretrained_vit import ViT
from dataset import ImageDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(100)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def str2bool(x):
    if x.lower() == 'true':
        return True
    elif x.lower() == 'false':
        return False
    else:
        raise ValueError('Invalid argument')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_dir', '-dt', type=str, default='hw3_data/p1_data/train/')
    parser.add_argument('--val_data_dir', '-dv', type=str, default='hw3_data/p1_data/val')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--model_name', '-m', type=str, default='B_16_imagenet1k')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', '-bs', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_epoch', type=int, default=5)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--warmup_ratio', type=float, default=0.2)
    args = parser.parse_args()
    assert args.log_interval % args.grad_step == 0
    print(args)

    checkpoint_dir = os.path.join('./result/p1', args.exp_name)
    log_dir = os.path.join(checkpoint_dir, 'log')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    train_transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         
         transforms.RandomHorizontalFlip(),
         transforms.RandomRotation(15), 
         transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)), 
         transforms.RandomPerspective(distortion_scale=0.2,p=0.5), 

         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    val_transform = transforms.Compose(
        [transforms.Resize((384, 384)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    train_dataset = ImageDataset(root=args.train_data_dir, transform=train_transform, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    val_dataset = ImageDataset(root=args.val_data_dir, transform=val_transform, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = ViT(args.model_name)
    model.load_state_dict(torch.load('pretrained_models/{}.pth'.format(args.model_name))) # torch 1.2.0 has bugs when using default load model function
    model.fc = nn.Linear(768, 37)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    
    if args.model_name == 'B_16_imagenet1k':
        model_size = 768
    else:
        raise ValueError('Invalid model name')
    warmup_steps = int(args.warmup_ratio * len(train_dataloader) * args.epoch / args.grad_step)
    noam_lambda = lambda step: (
            model_size ** (-0.5) * min((step + 1) ** (-0.5), (step + 1) * warmup_steps ** (-1.5)))
    scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'valid'))

    iteration = 0
    max_correct = 0
    total_iter = len(train_dataloader) * args.epoch
    temp_loss = 0
    temp_accuracy = 0
    for e in range(args.epoch):
        model.train()
        
        for i, (imgs, labels) in enumerate(train_dataloader):
            
            imgs, labels = imgs.to(device), labels.to(device)

            # source label loss
            logits = model(imgs)
            loss = criterion(logits, labels)
            predict = torch.argmax(logits, dim=-1)
            accuracy = torch.sum(predict == labels.view(predict.shape)).item() / imgs.shape[0]
            temp_loss += loss.item()
            temp_accuracy += accuracy

            loss = loss / args.grad_step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            if iteration % args.grad_step == 0:
                # update model
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if iteration % args.log_interval == 0:
                train_writer.add_scalar('loss', temp_loss / args.log_interval, iteration)
                train_writer.add_scalar('acc', temp_accuracy / args.log_interval, iteration)
                train_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
                    .format(e, args.epoch, iteration, total_iter, temp_loss / args.log_interval, temp_accuracy / args.log_interval))
                temp_loss = 0
                temp_accuracy = 0

            
            iteration += 1
        
        # validation
        model.eval()
        correct = 0
        temp_loss = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(val_dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                predict = torch.argmax(logits, dim=-1)
                correct += torch.sum(predict == labels.view(predict.shape)).item()
                temp_loss += loss.item()
        val_writer.add_scalar('loss', temp_loss / len(val_dataloader), iteration)
        val_writer.add_scalar('acc', correct / len(val_dataset), iteration)
        print('[VAL]\tEpoch [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'.format(e, args.epoch, temp_loss / len(val_dataloader), correct / len(val_dataset)))


        if e % args.save_epoch == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'vit_{}.pth'.format(e))
            save_checkpoint(checkpoint_path, model, optimizer)
        if correct >= max_correct:
            max_correct = correct
            checkpoint_path = os.path.join(checkpoint_dir, 'vit_best.pth')
            save_checkpoint(checkpoint_path, model, optimizer)


    checkpoint_path = os.path.join(checkpoint_dir, 'vit_{}.pth'.format(e))
    save_checkpoint(checkpoint_path, model, optimizer)



