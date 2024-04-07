

import torch.nn as nn
import torchvision


from PIL import Image
from numpy import asarray
import os
import numpy as np
import math
import copy
import argparse

import torch
from torchvision import datasets, transforms
import torch
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
import torchvision.datasets as datasets  # Standard datasets
import torchvision.transforms as transforms  # Transformations we can perform on our dataset for augmentation
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch import nn  # All neural network modules
from torch.utils.data import (
    DataLoader,Subset
)

from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import socket
socket.setdefaulttimeout(30)
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




def cnn_calc(n,p,f,s):
  return int(((math.floor(n + 2*p - f)/(s*1.0)) + 1))


def calculate_denseLayer(filter_seq,channel_list):
  padding = 1
  stride = 1
  hidden_count = 5
  initial_nodes_w = 256
  initial_nodes_b = 256

  for layers in range(1,hidden_count+1):
    initial_nodes_w = cnn_calc(initial_nodes_w,padding,filter_seq[layers-1],stride)
    initial_nodes_w = cnn_calc(initial_nodes_w,0,2,2)

    initial_nodes_b = cnn_calc(initial_nodes_b,padding,filter_seq[layers-1],stride)
    initial_nodes_b = cnn_calc(initial_nodes_b,0,2,2)

  lastConvNodes = initial_nodes_w*initial_nodes_b*channel_list[-1]
  return lastConvNodes

class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10,channel_list=[32,32,64,64,128],kernel = [3,3,3,3,3],fcLayerNodes = 192,dropOut = 0,activation_function = torch.nn.ReLU(),batchNorm='false'):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=channel_list[0],
            kernel_size=kernel[0],
            stride=1,
            padding=1,
        )
        self.dropout1 = nn.Dropout(dropOut)
        self.batchNorm1 = nn.BatchNorm2d(channel_list[0])
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=channel_list[0],
            out_channels=channel_list[1],
            kernel_size=kernel[1],
            stride=1,
            padding=1,
        )
        self.dropout2 = nn.Dropout(dropOut)
        self.batchNorm2 = nn.BatchNorm2d(channel_list[1])
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(
            in_channels=channel_list[1],
            out_channels=channel_list[2],
            kernel_size=kernel[2],
            stride=1,
            padding=1,
        )
        self.dropout3 = nn.Dropout(dropOut)
        self.batchNorm3 = nn.BatchNorm2d(channel_list[2])
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(
            in_channels=channel_list[2],
            out_channels=channel_list[3],
            kernel_size=kernel[3],
            stride=1,
            padding=1,
        )
        self.dropout4 = nn.Dropout(dropOut)
        self.batchNorm4 = nn.BatchNorm2d(channel_list[3])
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(
            in_channels=channel_list[3],
            out_channels=channel_list[4],
            kernel_size=kernel[4],
            stride=1,
            padding=1,
        )
        self.dropout5 = nn.Dropout(dropOut)
        self.batchNorm5 = nn.BatchNorm2d(channel_list[4])
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(calculate_denseLayer(kernel,channel_list), fcLayerNodes)
        self.dropout6 = nn.Dropout(dropOut)
        self.batchNorm6 = nn.BatchNorm1d(fcLayerNodes)
        self.fc2 = nn.Linear(fcLayerNodes, num_classes)
        self.activation = activation_function
        self.isBatchNorm = batchNorm



    def forward(self, x):
        if(self.isBatchNorm == 'false'):
            x = self.activation(self.conv1(x))
            x = self.dropout1(x)
            x = self.pool1(x)

            x = self.activation(self.conv2(x))
            x = self.dropout2(x)
            x = self.pool2(x)

            x = self.activation(self.conv3(x))
            x = self.dropout3(x)
            x = self.pool3(x)

            x = self.activation(self.conv4(x))
            x = self.dropout4(x)
            x = self.pool4(x)

            x = self.activation(self.conv5(x))
            x = self.dropout5(x)
            x = self.pool5(x)

            x = x.reshape(x.shape[0], -1)
            x = self.activation(self.fc1(x))
            x = self.dropout6(x)
            x = self.fc2(x)
        else:
            x = self.activation(self.batchNorm1(self.conv1(x)))
            x = self.dropout1(x)
            x = self.pool1(x)

            x = self.activation(self.batchNorm2(self.conv2(x)))
            x = self.dropout2(x)
            x = self.pool2(x)

            x = self.activation(self.batchNorm3(self.conv3(x)))
            x = self.dropout3(x)
            x = self.pool3(x)

            x = self.activation(self.batchNorm4(self.conv4(x)))
            x = self.dropout4(x)
            x = self.pool4(x)

            x = self.activation(self.batchNorm5(self.conv5(x)))
            x = self.dropout5(x)
            x = self.pool5(x)

            x = x.reshape(x.shape[0], -1)
            x = self.activation(self.batchNorm6(self.fc1(x)))
            x = self.dropout6(x)
            x = self.fc2(x)

        return x

def trainCnn(config,train_data,val_data,test_data):
  num_classes = 10
  input_channels = 3
  num_epochs = config.epochs #epochs
  filter_seq =  config.filter_customization
  activation = config.activation #activation
  dropout = config.dropout #dropout
  learning_rate = config.learning_rate #learning_rate
  batch_Size = config.batch_size #batch_size
  data_augmen = config.data_aug #data_aug
  denseLayerNodes = config.fc_layer_nodes #fc_layer_nodes
  fullyConnect = 2
  kernel_list = config.kernel_customization
  optimizer = config.optimizer
  augmentation = config.data_aug
  weightDecay = config.weight_decay
  isBatchNorm = config.batch_norm
  activation_fun = config.activation
  if(config.activation == 'relu'):
    activation_fun = torch.nn.ReLU()
  elif(config.activation == 'selu'):
    activation_fun = torch.nn.SELU()
  else:
    activation_fun = torch.nn.ELU()


  model = CNN( in_channels=input_channels, num_classes=num_classes,channel_list=filter_seq,kernel=kernel_list,fcLayerNodes=denseLayerNodes,dropOut = dropout,activation_function=activation_fun,batchNorm = isBatchNorm).to(device)

  criterion = nn.CrossEntropyLoss()
  if(optimizer == 'adam'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay = config.weight_decay)
  elif(optimizer == 'sgd'):
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,weight_decay = config.weight_decay)
  else:
    optimizer = optim.NAdam(model.parameters(), lr=learning_rate,weight_decay = config.weight_decay)


  for epoch in range(num_epochs):
      for batch_idx, (data, targets) in enumerate(tqdm(train_data)):
          data = data.to(device=device)
          targets = targets.to(device=device)

          scores = model(data)
          loss = criterion(scores, targets)

          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
      val_acc,val_loss = check_accuracy(val_data, model)
      train_acc,train_loss = check_accuracy(train_data, model)
      wandb.log({"Acc_val":val_acc,"Acc_val_loss":val_loss,"Acc_train":train_acc,"Acc_train_loss":train_loss,"epoch":epoch})

def check_accuracy(loader, model):
  num_correct = 0
  num_samples = 0
  total_loss = 0
  model.eval()

  with torch.no_grad():
      for x, y in loader:
          x = x.to(device=device)
          y = y.to(device=device)

          scores = model(x)
          loss = F.cross_entropy(scores, y)
          total_loss += loss.item()

          _, predictions = scores.max(1)
          num_correct += (predictions == y).sum().item()
          num_samples += predictions.size(0)

  accuracy = num_correct / num_samples
  average_loss = total_loss / len(loader)

  model.train()
  return accuracy, average_loss



def fitModel(args):
  resizeHeight = 256
  resizeWidth = 256

  transform = transforms.Compose([transforms.Resize((resizeHeight,resizeWidth)),
  transforms.ToTensor(),transforms.Lambda(lambda x: x/255.0)])


  augmen_transform = transforms.Compose([

      transforms.RandomHorizontalFlip(),
      transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),

  ])

  data_dir = '/inaturalist_12K/train/'
  train_dataset = datasets.ImageFolder(data_dir, transform=transform)



  # Define the percentage of validation data
  validation_ratio = 0.2  # 20%

  num_validation_samples = int(validation_ratio * len(train_dataset))

  train_size = len(train_dataset) - num_validation_samples
  train_dataset, val_dataset = random_split(train_dataset, [train_size, num_validation_samples])

  class_indices = [[] for _ in range(len(train_dataset.classes))]

  for i, (image, label) in enumerate(train_dataset):
      class_indices[label].append(i)

  train_indices = []
  val_indices = []
  Augtest_indices = []

  for indices in class_indices:
    n = len(indices)
    train_size = int(0.8 * n)
    val_size = int(0.2 * n)
    Augtest_size = n/5

    train_indices.extend(indices[:train_size])
    val_indices.extend(indices[train_size:train_size+val_size])
    Augtest_indices.extend(indices[:Augtest_size])

  new_train_dataset = Subset(train_dataset, train_indices)
  new_val_dataset = Subset(train_dataset, val_indices)
  Augtest_dataset = Subset(train_dataset, Augtest_indices)

  data_dir = '/inaturalist_12K/val'
  test_dataset = datasets.ImageFolder(data_dir, transform=transform)
  wandb.run.name = "hidden_" + str(args.epochs)+"filter_size"+str(args.filter_customization)+"_acc_"+ str(args.fc_layer_nodes)
  np.random.seed(1)

  train_data = torch.utils.data.DataLoader(train_dataset, batch_size=args.config.batch_size, shuffle=True,num_workers = 4, pin_memory=True)
  val_data = torch.utils.data.DataLoader(val_dataset, batch_size=args.config.batch_size, shuffle=True,num_workers = 4, pin_memory=True)

  test_data = torch.utils.data.DataLoader(test_dataset, batch_size=args.config.batch_size, shuffle=True, num_workers = 4,pin_memory=True)
  trainCnn(args,train_data,val_data,test_data)


def parse_arguments():
  parser = argparse.ArgumentParser(description='Training Parameters')
  parser.add_argument('-wp', '--wandb_project', type=str, default='AssignmentDL_2',
                        help='Project name')
  
  parser.add_argument('-we', '--wandb_entity', type=str, default='Entity_DL',
                        help='Wandb Entity')
  
  parser.add_argument('-e', '--epochs', type=int, default=10,help='Number of epochs for training network')

  parser.add_argument('-b', '--batch_size', type=int, default=64,help='Batch size for training neural network')

  
  parser.add_argument('-o', '--optimizer', type=str, default='nadam', choices = ["sgd","adam", "nadam"],help='Choice of optimizer')
   
  parser.add_argument('-lr', '--learning_rate', type=int, default=0.001, help='Learning rate')

  parser.add_argument('-w_d','--weight_decay',  type=float, default=0.0005, help='Weight decay parameter')

  parser.add_argument( '-a','--activation', type=str, default="relu",choices=[ "elu", "relu","selu","ELU"], help='activation functions')
  
  parser.add_argument( '-dp','--dropout', type=float, default=0,choices=[ 0,0.1,0.2], help='dropout values')

  parser.add_argument( '-da','--data_aug', type=str, default="false",choices=[ "true","false"], help='data augmentation')
  parser.add_argument( '-bn','--batch_norm', type=str, default="false",choices=[ "true","false"], help='data augmentation')

  parser.add_argument( '-n','--fc_layer_nodes', type=int, default=128,choices=[ 128,192,256,512], help='dense layer neurons')

  parser.add_argument( '-k','--kernel_customization', type=list, default=[3,3,3,3,3],choices=[[3,3,3,3,3],[3,3,5,5,7],[3,3,5,5,3],[3,5,7,5,5]], help='kernel sizes')
  parser.add_argument( '-fc','--filter_customization', type=list, default=[8,16,32,64,128],choices=[[8,16,32,64,128],[16,32,64,128,256],[32,64,128,256,512],[32,64,64,128,256],[16,64,96,256,460],[32,32,32,32,32]], help='no of filter in each layer')

  return parser.parse_args()

args = parse_arguments()

wandb.init(project=args.wandb_project)
fitModel(args)