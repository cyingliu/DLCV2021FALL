import sys
import os
import random
import argparse
import numpy as np
import torch
from torchvision import models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from byol_pytorch import BYOL

from dataset_p2 import MiniDataset

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

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir_train', '-dt', type=str, default='hw4_data/mini/train')
    parser.add_argument('--csv_path_train', '-pt', type=str, default='hw4_data/mini/train.csv')
    parser.add_argument('--data_dir_val', '-dv', type=str, default='hw4_data/mini/val')
    parser.add_argument('--csv_path_val', '-pv', type=str, default='hw4_data/mini/val.csv')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', '-bs', default=32)
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--save_epoch', type=int, default=20)
    args = parser.parse_args()
    print(args)

    checkpoint_dir = os.path.join('./result/p2/pretrain', args.exp_name)
    log_dir_train = os.path.join(checkpoint_dir, 'log', 'train')
    log_dir_val = os.path.join(checkpoint_dir, 'log', 'val')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir_train):
        os.makedirs(log_dir_train)
    if not os.path.exists(log_dir_val):
        os.makedirs(log_dir_val)

    train_dataset = MiniDataset(csv_path=args.csv_path_train, data_dir=args.data_dir_train) # 38400
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)  # 600

    val_dataset = MiniDataset(csv_path=args.csv_path_val, data_dir=args.data_dir_val) # 9600
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) # 150

    resnet = models.resnet50(pretrained=False)

    learner = BYOL(
        resnet,
        image_size = 128,
        hidden_layer = 'avgpool'
    )
    learner.net.to(device)

    optimizer = torch.optim.Adam(learner.parameters(), lr=args.lr)
    
    train_writer = SummaryWriter(log_dir=log_dir_train)
    val_writer = SummaryWriter(log_dir=log_dir_val)
    
    iteration = 0
    total_iter = len(train_dataloader) * args.epoch
    min_loss = float('inf')
    for e in range(args.epoch):
        learner.net.train()
        for images in train_dataloader:
            images = images.to(device)
            loss = learner(images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            learner.update_moving_average() # update moving average of target encoder
            if iteration % args.log_interval == 0:
                train_writer.add_scalar('loss', loss.item(), iteration)
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}'
                    .format(e, args.epoch, iteration, total_iter, loss.item()))
            iteration += 1
        
        learner.net.eval()
        temp_loss = 0
        with torch.no_grad():
            for images in val_dataloader:
                images = imgs.to(device)
                loss = learner(images)
                temp_loss += loss.item()
            avg_loss = temp_loss / len(val_dataloader)
            val_writer.add_scalar('loss', avg_loss, iteration)
            print('[VAL] Epoch [{}/{}]\tLoss: {:.4f}'
                .format(e, args.epoch, loss))
            if avg_loss < min_loss:
                min_loss = avg_loss
                checkpoint_path = os.path.join(checkpoint_dir, 'model-best.pth')
                save_checkpoint(checkpoint_path, learner.net, optimizer)
        if e % args.save_epoch == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'model-{}.pth'.format(e))
            save_checkpoint(checkpoint_path, learner.net, optimizer)





