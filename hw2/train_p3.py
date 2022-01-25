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

from dataset_p3 import DigitDataset
from model_p3 import DANN, DANN_small, DANN_tiny_bn_dp, DANN_small_domain3_alpha, DANN_micro

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
    parser.add_argument('--source_data_dir', '-ds', type=str, default='hw2_data/digits/mnistm/')
    parser.add_argument('--target_data_dir', '-d_target', type=str, default='hw2_data/digits/usps/')
    parser.add_argument('--test_data_dir', '-d_test', type=str, default='hw2_data/digits/usps/')
    parser.add_argument('--train_both', '-tb', type=str2bool, default=False)
    parser.add_argument('--exp_name', '-n', type=str, default='tmp')
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', '-bs', type=int, default=128)
    parser.add_argument('--epoch', '-e', type=int, default=10)
    parser.add_argument('--domain_ratio', type=float, default=1.0)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_epoch', type=int, default=20)
    parser.add_argument('--grad_step', type=int, default=1)
    args = parser.parse_args()
    print(args)

    checkpoint_dir = os.path.join('./result/p3', args.exp_name)
    log_dir = os.path.join(checkpoint_dir, 'log')
    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)

    transform = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ] )
    transform_usps = transforms.Compose(
        [transforms.Resize((28, 28)),
         transforms.ToTensor(),
         transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # for usps
         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ] )
    
    source_dataset = DigitDataset(args.source_data_dir, 
                                    transform if 'usps' not in args.source_data_dir else transform_usps, 
                                    mode='train', domain='source')
    source_dataloader = DataLoader(source_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    target_dataset = DigitDataset(args.target_data_dir, 
                                    transform if 'usps' not in args.target_data_dir else transform_usps, 
                                    mode='train', domain='target')
    target_dataloader = DataLoader(target_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    test_dataset = DigitDataset(args.test_data_dir, 
                                transform if 'usps' not in args.test_data_dir else transform_usps,
                                mode='test', domain=None)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = DANN_micro()
    model.to(device)

    criterion_domain = nn.BCELoss()
    criterion_class = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
    # optimizer = torch.optim.Adam( [
    #     {"params": model.feature_extractor.parameters(), "lr": 1e-4},
    #     {"params": model.domain_classifier.parameters(), "lr": 2e-4},
    #     {"params": model.label_classifier.parameters(), "lr": 2e-4},
    # ], lr=args.lr, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    train_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
    val_writer = SummaryWriter(log_dir=os.path.join(log_dir, 'valid'))

    iteration = 0
    total_iter = len(source_dataloader) * args.epoch
    for e in range(args.epoch):
        model.train()
        target_dataloader_iterator = iter(target_dataloader)
        # source_dataloader_iterator = iter(source_dataloader) ## usps
        for i, sources in enumerate(source_dataloader):
            try:
                targets = next(target_dataloader_iterator)
            except StopIteration:
                target_dataloader_iterator = iter(target_dataloader)
                targets = next(target_dataloader_iterator)
        # for i, targets in enumerate(target_dataloader):
        #     try:
        #         sources = next(source_dataloader_iterator)
        #     except StopIteration:
        #         source_dataloader_iterator = iter(source_dataloader)
        #         sources = next(source_dataloader_iterator)
            
            source_imgs, source_class_labels = sources
            target_imgs = targets
            source_imgs, source_class_labels, target_imgs = source_imgs.to(device), source_class_labels.to(device), target_imgs.to(device)
            bs_source = source_imgs.shape[0]
            bs_target = target_imgs.shape[0]

            p = float(iteration + e * len(source_dataloader)) / \
                args.epoch / len(source_dataloader)
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # source label loss
            source_class_logits, source_domain_logits = model(source_imgs, alpha)
            source_class_loss = criterion_class(source_class_logits, source_class_labels)
            predict = torch.argmax(source_class_logits, dim=-1)
            accuracy = torch.sum(predict == source_class_labels.view(predict.shape)).item() / bs_source

            if args.train_both:
                source_domain_labels = torch.ones((bs_source, 1)).to(device)
                target_domain_labels = torch.zeros((bs_target, 1)).to(device)
                target_class_logits, target_domain_logits = model(target_imgs, alpha)
                source_domain_loss = criterion_domain(source_domain_logits, source_domain_labels)
                target_domain_loss = criterion_domain(target_domain_logits, target_domain_labels)

            if args.train_both:
                loss = (source_class_loss + args.domain_ratio * (source_domain_loss + target_domain_loss)) * 3 / (1 + 2 * args.domain_ratio)
            else:
                loss = source_class_loss
            # update model
            model.zero_grad()
            loss.backward()
            optimizer.step()
            
            if iteration % args.log_interval == 0:
                train_writer.add_scalar('class_loss', source_class_loss.item(), iteration)
                train_writer.add_scalar('acc_class', accuracy, iteration)
                if args.train_both:
                    train_writer.add_scalar('source_domain_loss', source_domain_loss.item(), iteration)
                    train_writer.add_scalar('target_domain_loss', target_domain_loss.item(), iteration)
                print('Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'.format(e, args.epoch, iteration, total_iter, source_class_loss.item(), accuracy))

            iteration += 1
        # scheduler.step()
        # validation
        model.eval()
        correct = 0
        loss = 0
        with torch.no_grad():
            for i, (imgs, labels) in enumerate(test_dataloader):
                imgs, labels = imgs.to(device), labels.to(device)
                class_logit, domain_logit = model(imgs, alpha)
                loss_class = criterion_class(class_logit, labels)
                predict = torch.argmax(class_logit, dim=-1)
                correct += torch.sum(predict == labels.view(predict.shape)).item()
                loss += loss_class.item()
        val_writer.add_scalar('class_loss', loss / len(test_dataloader), iteration)
        val_writer.add_scalar('acc_class', correct / len(test_dataset), iteration)
        print('[VAL]\tEpoch [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'.format(e, args.epoch, loss / len(test_dataloader), correct / len(test_dataset)))


        if e % args.save_epoch == 0:
            checkpoint_path = os.path.join(checkpoint_dir, 'dann_{}.pth'.format(e))
            save_checkpoint(checkpoint_path, model, optimizer)


    checkpoint_path = os.path.join(checkpoint_dir, 'dann_{}.pth'.format(e))
    save_checkpoint(checkpoint_path, model, optimizer)



