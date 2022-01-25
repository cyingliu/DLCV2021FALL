import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import argparse

from dataset import MiniDataset
from sampler import PrototypicalSampler
from torch.utils.data import DataLoader
from model import Convnet
from utils import euclidean_metric, cosine_similarity_metric, parametric_metric

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

def save_checkpoint(checkpoint_path, model, optimizer, parametric_metric_model):
    if parametric_metric_model == None:
      state = {'state_dict': model.state_dict(), 'opt': optimizer.state_dict()}
    else:
      state = {'state_dict': model.state_dict(), 'opt': optimizer.state_dict(), 'state_dict_parametric': parametric_metric_model.state_dict()}
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
  parser.add_argument('--n_shot_train', '-nst', type=int, default=1)
  parser.add_argument('--n_way_train', '-nwt', type=int, default=5)
  parser.add_argument('--n_query_train', '-nqt', type=int, default=15)
  parser.add_argument('--n_iter_train', '-nit', type=int, default=100)
  parser.add_argument('--n_shot_val', '-nsv', type=int, default=1)
  parser.add_argument('--n_way_val', '-nwv', type=int, default=5)
  parser.add_argument('--n_query_val', '-nqv', type=int, default=75)
  parser.add_argument('--n_iter_val', '-niv', type=int, default=100)
  parser.add_argument('--dist', '-d', type=str, default='euclidean') # ['euclidean', 'cosine', 'parametric']
  parser.add_argument('--log_interval', type=int, default=25)
  parser.add_argument('--save_epoch', type=int, default=20)
  args = parser.parse_args()
  print(args)
  assert args.dist == 'euclidean' or args.dist == 'cosine' or args.dist == 'parametric'

  checkpoint_dir = os.path.join('./result/p1', args.exp_name)
  log_dir_train = os.path.join(checkpoint_dir, 'log', 'train')
  log_dir_val = os.path.join(checkpoint_dir, 'log', 'val')
  if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
  if not os.path.exists(log_dir_train):
    os.makedirs(log_dir_train)
  if not os.path.exists(log_dir_val):
    os.makedirs(log_dir_val)
    

  def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

  train_dataset = MiniDataset(csv_path=args.csv_path_train, data_dir=args.data_dir_train)
  train_loader = DataLoader(
      train_dataset,
      num_workers=2, pin_memory=False, worker_init_fn=worker_init_fn,
      batch_sampler=PrototypicalSampler(label=train_dataset.data_df['label'].tolist(), 
                                  iteration=args.n_iter_train, class_per_it=args.n_way_train, sample_per_class=args.n_shot_train+args.n_query_train))
  val_dataset = MiniDataset(csv_path=args.csv_path_val, data_dir=args.data_dir_val)
  val_loader = DataLoader(
      val_dataset,
      num_workers=2, pin_memory=False, worker_init_fn=worker_init_fn,
      batch_sampler=PrototypicalSampler(label=val_dataset.data_df['label'].tolist(),
                      iteration=args.n_iter_val, class_per_it=args.n_way_val, sample_per_class=args.n_shot_val+args.n_query_val)
      )

  model = Convnet()
  model.to(device)
  if args.dist == 'parametric':
    assert args.n_way_train == args.n_way_val
    parametric_metric_model = parametric_metric(args.n_way_train)
    parametric_metric_model.to(device)
    optimizer_parametric = torch.optim.Adam(parametric_metric_model.parameters(), lr=args.lr, betas=(0.9, 0.999))
  
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
  criterion = nn.CrossEntropyLoss()

  train_writer = SummaryWriter(log_dir=log_dir_train)
  val_writer = SummaryWriter(log_dir=log_dir_val)

  total_iter = args.epoch * args.n_iter_train
  iteration = 0
  max_acc = 0
  for e in range(args.epoch):
    model.train()
    if args.dist == 'parametric': 
        parametric_metric_model.train()
    for i, (img, label) in enumerate(train_loader):
      img = img.to(device)
      data_support, data_query = img[:args.n_way_train * args.n_shot_train], img[args.n_way_train * args.n_shot_train:]
      
      # create the relative label (0 ~ N_way-1) for query data
      label_encoder = {label[i] : i for i in range(args.n_way_train)}      
      query_label = torch.LongTensor([label_encoder[class_name] for class_name in label[args.n_way_train * args.n_shot_train:]]).to(device)
    
      prototype = model(data_support) 
      prototype = prototype.reshape(args.n_shot_train, args.n_way_train, -1).mean(dim=0) # (3, 1600)

      output = model(data_query)
      if args.dist == 'euclidean':
          logits = euclidean_metric(output, prototype)
      elif args.dist == 'cosine':
          logits = cosine_similarity_metric(output, prototype)
      elif args.dist == 'parametric':
          logits = parametric_metric_model(output, prototype)
      loss = criterion(logits, query_label)

      predict = torch.argmax(logits, dim=-1)
      accuracy = torch.sum(predict == query_label.view(predict.shape)).item() / data_query.shape[0]

      # update model
      optimizer.zero_grad()
      if args.dist == 'parametric':
        optimizer_parametric.zero_grad()
      loss.backward()
      optimizer.step()
      if args.dist == 'parametric':
        optimizer_parametric.step()

      if iteration % args.log_interval == 0:
        train_writer.add_scalar('loss', loss.item(), iteration)
        train_writer.add_scalar('acc', accuracy, iteration)
        print('Epoch [{}/{}]\tIter [{}/{}]\tLoss: {:.4f}\tAcc: {:.4f}'
          .format(e, args.epoch, iteration, total_iter, loss.item(), accuracy))
      iteration += 1
    
    # validation
    model.eval()
    if args.dist == 'parametric':
      parametric_metric_model.eval()
    temp_loss = 0
    temp_correct = 0
    with torch.no_grad():
      for i, (img, label) in enumerate(val_loader):
        img = img.to(device)
        data_support, data_query = img[:args.n_way_val * args.n_shot_val], img[args.n_way_val * args.n_shot_val:]
        # create the relative label (0 ~ N_way-1) for query data
        label_encoder = {label[i] : i for i in range(args.n_way_val)}
        query_label = torch.LongTensor([label_encoder[class_name] for class_name in label[args.n_way_val * args.n_shot_val:]]).to(device)
      
        prototype = model(data_support) 
        prototype = prototype.reshape(args.n_shot_val, args.n_way_val, -1).mean(dim=0) # (3, 1600)

        output = model(data_query)
        if args.dist == 'euclidean':
          logits = euclidean_metric(output, prototype)
        elif args.dist == 'cosine':
          logits = cosine_similarity_metric(output, prototype)
        elif args.dist == 'parametric':
          logits = parametric_metric_model(output, prototype)
        loss = criterion(logits, query_label)
        temp_loss += loss.item()

        predict = torch.argmax(logits, dim=-1)
        correct = torch.sum(predict == query_label.view(predict.shape)).item()
        temp_correct += correct
    
    loss = temp_loss / len(val_loader)
    accuracy = temp_correct / (args.n_query_val * args.n_way_val * args.n_iter_val)
    val_writer.add_scalar('loss', loss, iteration)
    val_writer.add_scalar('acc', accuracy, iteration)
    print('[VAL] Epoch [{}/{}]\tLoss: {:.4f}\tAcc:{:.4f}'
        .format(e, args.epoch, loss, accuracy))
    if e % args.save_epoch == 0:
      checkpoint_path = os.path.join(checkpoint_dir, 'model-{}.pth'.format(e))
      save_checkpoint(checkpoint_path, model, optimizer, parametric_metric_model if args.dist == 'parametric' else None)
    if accuracy >= max_acc:
      checkpoint_path = os.path.join(checkpoint_dir, 'model-best.pth')
      save_checkpoint(checkpoint_path, model, optimizer, parametric_metric_model if args.dist == 'parametric' else None)
      max_acc = accuracy

  checkpoint_path = os.path.join(checkpoint_dir, 'model-{}.pth'.format(e))
  save_checkpoint(checkpoint_path, model, optimizer, parametric_metric_model if args.dist == 'parametric' else None)






