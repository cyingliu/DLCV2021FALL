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

from model_p2 import Discriminator_fmnist, Generator_fmnist, Discriminator, Generator, Generator_big, Discriminator_small, Discriminator_big, Generator_64, Discriminator_64, Generator_32, Discriminator_32
from dataset_p2 import MNISTMDataset

# Use GPU if available, otherwise stick with cpu
use_cuda = torch.cuda.is_available()
torch.manual_seed(123)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device used:', device)

def save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D):
    state = {'state_dict_G': model_G.state_dict(), 'state_dict_D': model_D.state_dict(), 
    'opt_G': optimizer_G.state_dict(), 'opt_D': optimizer_D.state_dict()}
    torch.save(state, checkpoint_path)
    print('model saved to %s' % checkpoint_path)

def label2onehot(label):
    # print(label.shape, label)
    label_onehot = torch.zeros(label.shape[0], 10).to(device)
    label_onehot = label_onehot.scatter_(1, label, 1).view(label.shape[0], 10)
    return label_onehot

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', '-d', type=str, default='hw2_data/digits/mnistm/')
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--model_type', type=str, choices=['symmetric', 'fmnist'], default='fmnist')
    parser.add_argument('--G_lr', type=float, default=2e-4)
    parser.add_argument('--D_lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--G_step', type=int, default=1)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_epoch', type=int, default=10)
    parser.add_argument('--grad_step', type=int, default=1)
    parser.add_argument('--z_dim', type=int, default=100)
    parser.add_argument('--dim', type=int, default=64, help='dim for base num of channels for G and D') # symmetric DCGAN
    parser.add_argument('--G_emb_dim', type=int, default=50) # fmnist dcgan, symmetric dcgan
    parser.add_argument('--G_latent_dim', type=int, default=384) # fmnist dcgan, symmetric dcgan
    parser.add_argument('--G_n_channel', type=int, default=192) # fmnist dcgan
    parser.add_argument('--D_n_channel', type=int, default=16) # fmnist dcgan
    parser.add_argument('--D_class_ratio', type=float, default=1.0)
    parser.add_argument('--G_class_ratio', type=float, default=1.0)
    args = parser.parse_args()
    print(args)

    checkpoint_dir = os.path.join('./result/p2', args.exp_name)
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
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ] )
    train_dataset = MNISTMDataset(args.data_dir, transform, 'train')
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    if args.model_type == 'symmetric':
        model_G = Generator(in_dim=args.z_dim, dim=args.dim, emb_dim=args.G_emb_dim, latent_dim=args.G_latent_dim) 
        model_D = Discriminator(dim=args.dim)
    elif args.model_type == 'fmnist':
        # model_G = Generator_fmnist(emb_dim=args.G_emb_dim, latent_dim=args.G_latent_dim, n_channel=args.G_n_channel) 
        model_G = Generator_64()
        # model_D = Discriminator_fmnist(n_channel=args.D_n_channel) 
        model_D = Discriminator_64() 
    else: raise ValueError('Invalid model type')

    model_G.to(device)
    model_D.to(device)

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

    criterion_dis = nn.BCELoss()
    criterion_cla = nn.CrossEntropyLoss()
    optimizer_G = torch.optim.Adam(model_G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))
    # scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=40, gamma=0.5)
    # scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=40, gamma=0.5)
    writer = SummaryWriter(log_dir=log_dir)

    z_sample = torch.randn(100, args.z_dim).to(device)
    sample_label = torch.cat([torch.arange(10) for i in range(10)], dim=0).long().unsqueeze(-1).to(device)
    sample_label_onehot = label2onehot(sample_label).long()
    iteration = 0
    total_iter = len(train_dataloader) * args.epoch
    for e in range(args.epoch):
        for i, (img, label) in enumerate(train_dataloader):

            # ====================== train D =============================== #
            model_D.train()
            bs = img.shape[0]
            z = torch.randn(bs, args.z_dim).to(device)
            rand_label = torch.randint(0, 10, (bs, 1)).long().to(device)
            rand_label_onehot = label2onehot(rand_label).long()
            real_imgs = img.to(device)
            label = label.to(device)
            fake_imgs = model_G(z, rand_label_onehot)

            # label        
            real_label = torch.ones((bs)).to(device)
            fake_label = torch.zeros((bs)).to(device)

            # discriminator classify
            real_logit, real_class_logit = model_D(real_imgs.detach())
            fake_logit,  fake_class_logit = model_D(fake_imgs.detach())
            
            # compute dis loss
            real_loss = criterion_dis(real_logit, real_label)
            fake_loss = criterion_dis(fake_logit, fake_label)
            loss_dis_D = (real_loss + fake_loss) / 2
            fake_acc = np.mean(((fake_logit > 0.5).cpu().data.numpy() == fake_label.cpu().data.numpy()))
            real_acc = np.mean(((real_logit > 0.5).cpu().data.numpy() == real_label.cpu().data.numpy()))
            acc_dis_D = (real_acc + fake_acc) / 2
            # compute class loss
            real_class_loss = criterion_cla(real_class_logit, label)
            fake_class_loss = criterion_cla(fake_class_logit, rand_label.view(-1))
            loss_class_D = (real_class_loss + fake_class_loss) / 2

            loss_D = (loss_dis_D + args.D_class_ratio * loss_class_D) / (1 + args.D_class_ratio)

            # update model
            model_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # ====================== train G =============================== #
            model_G.train()
            for _ in range(args.G_step):
                z = torch.randn(bs, args.z_dim).to(device)
                rand_label = torch.randint(0, 10, (bs, 1)).long().to(device)
                rand_label_onehot = label2onehot(rand_label).long()
                fake_imgs = model_G(z, rand_label_onehot)

                # discriminator classify
                fake_logit, class_logit = model_D(fake_imgs)
                
                # compute loss
                loss_dis_G = criterion_dis(fake_logit, real_label)
                loss_cla_G = criterion_cla(class_logit, rand_label.view(-1))
                loss_G = (loss_dis_G + args.G_class_ratio * loss_cla_G) / (1 + args.G_class_ratio)
                # update model
                model_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

            if iteration % args.log_interval == 0:
                writer.add_scalar('loss_D', loss_D.item(), iteration)
                writer.add_scalar('loss_D_real', real_loss.item(), iteration)
                writer.add_scalar('loss_D_fake', fake_loss.item(), iteration)
                writer.add_scalar('loss_dis_D', loss_dis_D.item(), iteration)
                writer.add_scalar('loss_class_D', loss_class_D.item(), iteration)
                writer.add_scalar('acc_dis_D', acc_dis_D.item(), iteration)

                writer.add_scalar('loss_G', loss_G.item(), iteration)
                writer.add_scalar('loss_dis_G', loss_dis_G.item(), iteration)
                writer.add_scalar('loss_cla_G', loss_cla_G.item(), iteration)

                writer.add_scalar('lr', optimizer_G.param_groups[0]['lr'], iteration)
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss_D: {:.4f}\tLoss_G: {:.4f}'.format(e, args.epoch, iteration, total_iter, loss_D.item(), loss_G.item()))

            iteration += 1
        # scheduler_G.step()
        # scheduler_D.step()
        # log
        model_G.eval()
        with torch.no_grad():
            fake_imgs_sample = (model_G(z_sample, sample_label_onehot).data + 1) / 2.0
        filename = os.path.join(image_dir, 'Epoch_{}.jpg'.format(e))
        torchvision.utils.save_image(fake_imgs_sample, filename, nrow=10)
        print('\t| Save some samples to {}.'.format(filename))
        # show generated image
        model_G.train()
    
        if e % args.save_epoch == 0:
            print('\t| Save model to {}'.format(checkpoint_dir))
            checkpoint_path = os.path.join(checkpoint_dir, 'dcgan_{}.pth'.format(e))
            save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D)
    
    # save last model
    print('\t| Save model to {}'.format(checkpoint_dir))
    save_checkpoint(checkpoint_path, model_G, model_D, optimizer_G, optimizer_D)


