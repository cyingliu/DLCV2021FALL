import sys
import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset_p2 import OfficeDataset
from PIL import Image
filenameToPILImage = lambda x: Image.open(x)

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(SEED)
np.random.seed(SEED)

def save_checkpoint(checkpoint_path, model, optimizer):
    state = {'state_dict': model.state_dict(), 'opt': optimizer.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path)
    resnet_state = dict()
    for k, v in state['state_dict'].items():
        if k.startswith('net.'):
            resnet_state[k.replace('net.', '')] = v
    model.load_state_dict(resnet_state)
    print('model loaded from %s' % checkpoint_path)

def str2bool(x):
    assert x.lower() == 'true' or x.lower() == 'false'
    if x.lower() == 'true':
        return True
    else:
        return False
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_train', '-dt', type=str, default='hw4_data/office/train')
    parser.add_argument('--csv_path_train', '-pt', type=str, default='hw4_data/office/train.csv')
    parser.add_argument('--data_dir_val', '-dv', type=str, default='hw4_data/office/val')
    parser.add_argument('--csv_path_val', '-pv', type=str, default='hw4_data/office/val.csv')
    parser.add_argument('--checkpoint_path', '-ckp', type=str, default='result/p2/pretrain/baseline/learner-57-new.pth', help='pretrained checkpoint path of learner')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--fixbackbone', type=str2bool, default=False)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-bs', default=64)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--save_epoch', type=int, default=20)
    args = parser.parse_args()
    print(args)
    
    checkpoint_dir = os.path.join('./result/p2/finetune', args.exp_name)
    log_dir_train = os.path.join(checkpoint_dir, 'log', 'train')
    log_dir_val = os.path.join(checkpoint_dir, 'log', 'val')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_val):
        os.makedirs(log_dir_val)

    train_transform = transforms.Compose([
            filenameToPILImage,
            transforms.Resize((128, 128)),

            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(15), 
            # transforms.ColorJitter(brightness = (0.5, 1.5), contrast = (0.5, 1.5), saturation = (0.5, 1.5)), 
            # transforms.RandomPerspective(distortion_scale=0.2,p=0.5), 
            
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
                filenameToPILImage,
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

    train_dataset = OfficeDataset(csv_path=args.csv_path_train, data_dir=args.data_dir_train, 
                                    label2id_path='label2id.csv', transform=train_transform, mode='train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = OfficeDataset(csv_path=args.csv_path_val, data_dir=args.data_dir_val, 
                                    label2id_path='label2id.csv', transform=val_transform, mode='val')
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = models.resnet50(pretrained=False)
    if 'pretrain_model_SL_new.pt' in args.checkpoint_path:
        print('load SL model')
        model.load_state_dict(torch.load(args.checkpoint_path)) # 'result/p2/pretrain/pretrain_model_SL_new.pt'
    elif args.checkpoint_path == 'None':
        pass
    else:
        load_checkpoint(args.checkpoint_path, model)
    model.fc = nn.Linear(2048, 65)
    # freeze part of network
    if args.fixbackbone:
        for param in model.parameters():
            param.requires_grad = False
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
    model.to(device)

    if args.fixbackbone:
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    
    train_writer = SummaryWriter(log_dir=log_dir_train)
    val_writer = SummaryWriter(log_dir=log_dir_val)

    criterion = nn.CrossEntropyLoss()

    iteration = 0
    total_iter = len(train_dataloader) * args.epoch
    max_acc = 0
    for e in range(args.epoch):
        model.train()
        for images, labels in train_dataloader:
            
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            predict = torch.argmax(logits, dim=-1)
            accuracy = torch.sum(predict == labels.view(predict.shape)).item() / labels.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % args.log_interval == 0:
                train_writer.add_scalar('loss', loss.item(), iteration)
                train_writer.add_scalar('acc', accuracy, iteration)
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
                    .format(e, args.epoch, iteration, total_iter, loss.item(), accuracy))
            iteration += 1

        scheduler.step()
        
        model.eval()
        temp_loss = 0
        temp_correct = 0
        with torch.no_grad():
            for images, labels in val_dataloader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)

                predict = torch.argmax(logits, dim=-1)
                correct = torch.sum(predict == labels.view(predict.shape)).item()

                temp_loss += loss.item()
                temp_correct += correct

        avg_loss = temp_loss / len(val_dataloader)
        avg_acc = temp_correct / len(val_dataset)
        val_writer.add_scalar('loss', avg_loss, iteration)
        val_writer.add_scalar('acc', avg_acc, iteration)
        print('[VAL] Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
                .format(e, args.epoch, iteration, total_iter, avg_loss, avg_acc))
        if avg_acc > max_acc:
            max_acc = avg_acc
            checkpoint_path = os.path.join(checkpoint_dir, 'model-best.pth')
            save_checkpoint(checkpoint_path, model, optimizer)
        if e % args.save_epoch == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'model-{}.pth'.format(e))
            save_checkpoint(checkpoint_path, model, optimizer)


