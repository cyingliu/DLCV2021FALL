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

from model_p1 import Discriminator_wgan, Generator
from dataset_p1 import FaceDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D):
    state = {'state_dict_G': model_G.state_dict(), 'state_dict_D': model_D.state_dict(), 'opt_G': optimizer_G.state_dict(), 'opt_D': optimizer_D.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def load_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D):
    state = torch.load(checkpoint_path)
    model_G.load_state_dict(state['state_dict_G'])
    model_D.load_state_dict(state['state_dict_D'])
    optimizer_G.load_state_dict(state['opt_G'])
    optimizer_D.load_state_dict(state['opt_D'])
    for state in optimizer_G.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    for state in optimizer_D.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)
    print('model loaded from %s' % checkpoint_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='hw2_data/face/train')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--G_lr', type=float, default=5e-5)
    parser.add_argument('--D_lr', type=float, default=5e-5)
    parser.add_argument('--D_step', type=int, default=5)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--batch_size', '-bs', type=int, default=64)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=-1)
    parser.add_argument('--checkpoint_path', type=str, default='None')
    parser.add_argument('--log_interval', type=int, default=25)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=100)
    args = parser.parse_args()
    assert args.log_interval % args.D_step == 0
    print(args)


    checkpoint_dir = os.path.join('./result/p1', args.exp_name)
    log_dir = os.path.join(checkpoint_dir, 'log')
    image_dir = os.path.join(checkpoint_dir, 'image')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    if not os.path.exists(image_dir):
      os.makedirs(image_dir)

    transform = transforms.Compose(
        [transforms.Resize((64, 64)),
         # transforms.RandomHorizontalFlip(p=0.5),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3) ] )
    train_dataset = FaceDataset(args.data_dir, transform)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model_G = Generator(in_dim=args.z_dim)
    model_D = Discriminator_wgan()
    

    # Weight initialization
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    model_G.apply(weights_init)
    model_D.apply(weights_init)

    criterion = nn.BCELoss()

    optimizer_G = torch.optim.RMSprop(model_G.parameters(), lr=args.G_lr)
    optimizer_D = torch.optim.RMSprop(model_D.parameters(), lr=args.D_lr)
    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=40, gamma=0.5)
    # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=40, gamma=0.5)
    if args.start_epoch > 0:
        load_checkpoint(args.checkpoint_path, model_G, model_D, optimizer_G, optimizer_D)
    
    optimizer_G.param_groups[0]['lr'] /= 2
    optimizer_D.param_groups[0]['lr'] /= 2
    
    writer = SummaryWriter(log_dir=log_dir)

    z_sample = torch.randn(100, args.z_dim, 1, 1).to(device)
    if args.start_epoch > 0:
        iteration = (args.start_epoch - 1) * len(train_dataloader)
    else:
        iteration = 0
    total_iter = len(train_dataloader) * args.epoch
    model_G.to(device)
    model_D.to(device)
    for e in range(args.start_epoch, args.epoch):
        for i, data in enumerate(train_dataloader):

            # ====================== train D =============================== #
            
            model_D.train()
            bs = data.shape[0]
            z = torch.randn(bs, args.z_dim, 1, 1).to(device)
            real_imgs = data.to(device)
            fake_imgs = model_G(z)

            # label        
            real_label = torch.ones((bs)).to(device)
            fake_label = torch.zeros((bs)).to(device)
            # soft_real_label = (real_label * 0.9).to(device)

            # discriminator classify
            real_logit = model_D(real_imgs.detach())
            fake_logit = model_D(fake_imgs.detach())
            
            # compute loss
            real_loss = - torch.mean(real_logit)
            fake_loss = torch.mean(fake_logit)
            loss_D = (real_loss + fake_loss) / 2
            fake_acc = np.mean(((fake_logit > 0).cpu().data.numpy() == fake_label.cpu().data.numpy()))
            real_acc = np.mean(((real_logit > 0).cpu().data.numpy() == real_label.cpu().data.numpy()))
            acc_D = (real_acc + fake_acc) / 2

            # update model
            model_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            for p in model_D.parameters():
                p.data.clamp_(-args.clip_value, args.clip_value)

            # ====================== train G =============================== #
            if iteration % args.D_step == 0:
                model_G.train()
                z = torch.randn(bs, args.z_dim, 1, 1).to(device)
                fake_imgs = model_G(z)

                # discriminator classify
                fake_logit = model_D(fake_imgs)
                
                # compute loss
                # loss_G = criterion(fake_logit, real_label)
                loss_G = - torch.mean(fake_logit)

                # update model
                model_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            if iteration % args.log_interval == 0:
                writer.add_scalar('loss_D', loss_D.item(), iteration)
                writer.add_scalar('loss_D_real', real_loss.item(), iteration)
                writer.add_scalar('loss_D_fake', fake_loss.item(), iteration)
                writer.add_scalar('acc_D', acc_D.item(), iteration)
                writer.add_scalar('loss_G', loss_G.item(), iteration)
                writer.add_scalar('lr', optimizer_G.param_groups[0]['lr'], iteration)
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}'.format(e, args.epoch, iteration, total_iter, loss_D.item(), loss_G.item()))

            iteration += 1
        # scheduler_G.step()
        # scheduler_D.step()
        # log
        model_G.eval()
        with torch.no_grad():
            fake_imgs_sample = (model_G(z_sample).data + 1) / 2.0
        filename = os.path.join(image_dir, 'Epoch_{}.jpg'.format(e))
        torchvision.utils.save_image(fake_imgs_sample, filename, nrow=10)
        print('\t| Save some samples to {}.'.format(filename))
        # show generated image
        model_G.train()
    
        if e % args.save_epoch == 0:
            print('\t| Save model to {}'.format(checkpoint_dir))
            checkpoint_path = os.path.join(checkpoint_dir, 'dcgan_{}.pth'.format(e))
            save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D)
            # torch.save(model_G.state_dict(), os.path.join(checkpoint_dir, 'dcgan_g_{}.pth'.format(e)))
            # torch.save(model_D.state_dict(), os.path.join(checkpoint_dir, 'dcgan_d_{}.pth'.format(e)))
    
    # save last model
    print('\t| Save model to {}'.format(checkpoint_dir))
    checkpoint_path = os.path.join(checkpoint_dir, 'dcgan_{}.pth'.format(e))
    save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D)


