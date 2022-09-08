from audioop import bias
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys
import time
from enviroment.firegrid_v4 import FireGrid_V4
import copy

# Red estilo pytorch
class CNN(torch.nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(1,1), stride=1, padding = 3, bias = True)
    self.max_p1 = nn.MaxPool2d(3, stride=1)
    self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=(2,2), stride=1, padding = 3, bias = True)
    self.max_p2 = nn.MaxPool2d(2, stride=2)
    self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(5,5), bias = True)
    self.max_p3 = nn.MaxPool2d(5, stride=3)

    # FCN para policy
    self.linear1 = nn.Linear(1536, 256)
    self.linear2 = nn.Linear(256, 64)
    self.linear3 = nn.Linear(64, 16)

    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear3.weight, mode='fan_in', nonlinearity='relu')

  # Computa la pasada hacia adelante
  def forward(self, x):
  # Forward común
    u1 = self.conv1(x)
    h1 = F.relu(u1)
    f1 = self.max_p1(h1)
    # print(f1.shape)
    u2 = self.conv2(f1)
    h2 = F.relu(u2)
    f2 = self.max_p2(h2)
    u3 = self.conv3(f2)
    h3 = F.relu(u3)
    f3 = self.max_p3(h3)
    m = torch.flatten(input = f3)
    # Forward Policy
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    h4 = F.relu(u4)
    u5 = self.linear3(h4)
    y_pred = F.softmax(u5)
    return y_pred

# |net = CNN()
# env = FireGrid_V4(20, burn_value=10, n_sims=50)
# states = []
# state = env.reset()
# states.append(state)
# for i in range(10):
#     policy, value = net.forward(copy.copy(state))
#     action = policy.multinomial(1)
#     next_state, reward, done = env.step(action.detach())
# x, y = net.forward(state)
# print(x.shape)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)