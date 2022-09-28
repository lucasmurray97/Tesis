from audioop import bias
from ssl import ALERT_DESCRIPTION_UNKNOWN_PSK_IDENTITY
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import time
from enviroment.firegrid_v4 import FireGrid_V4
import copy
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.optim import AdamW
import datetime
from tqdm import tqdm
# Red estilo pytorch
class CNN(torch.nn.Module):
  def __init__(self, batch = False):
    super(CNN, self).__init__()
    self.batch = batch
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=(2,2), stride=2, padding = 0, bias = True)
    # self.max_p1 = nn.MaxPool2d(2, stride=1)
    self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.max_p2 = nn.MaxPool2d(2, stride=2)
    self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), padding = 1, bias = True)
    self.max_p3 = nn.MaxPool2d(2, stride=3)

    # FCN para policy
    self.linear1 = nn.Linear(256, 128)
    self.linear2 = nn.Linear(128, 64)
    self.linear3 = nn.Linear(64, 16)
    # FCN para value
    self.linear_1 = nn.Linear(256, 128)
    self.linear_2 = nn.Linear(128, 64)
    self.linear_3 = nn.Linear(64, 1)

    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear3.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear_1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear_2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear_3.weight, mode='fan_in', nonlinearity='relu')

  # Computa la pasada hacia adelante
  def forward(self, x):
  # Forward común
    u1 = self.conv1(x)
    h1 = F.relu(u1)
    # print(h1.shape)
    # f1 = self.max_p1(h1)
    # print(f1.shape)
    u2 = self.conv2(h1)
    h2 = F.relu(u2)
    # print(h2.shape)
    f2 = self.max_p2(h2)
    # print(f2.shape)
    u3 = self.conv3(f2)
    h3 = F.relu(u3)
    # print(h3.shape)
    f3 = self.max_p3(h3)
    # print(f3.shape)
    m = torch.flatten(input = f3, start_dim = 1)
    # print(m.shape)
    # Forward Policy
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    h4 = F.relu(u4)
    u5 = self.linear3(h4)
    y_pred = F.softmax(u5)
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    u_4 = self.linear_2(h_3)
    h_4 = F.relu(u_4)
    value_pred = self.linear_3(h_4)
    return y_pred, value_pred

def env2_step(step):
  state = torch.rand(3, 20, 20)
  reward = torch.rand(1)
  if step == 100:
    return state, reward, True
  return state, reward, False