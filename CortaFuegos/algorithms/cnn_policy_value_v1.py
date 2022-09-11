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
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import datetime
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
    self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=(5,5), padding = 1, bias = True)
    self.max_p3 = nn.MaxPool2d(7, stride=3)

    # FCN para policy
    self.linear1 = nn.Linear(1536, 256)
    self.linear2 = nn.Linear(256, 64)
    self.linear3 = nn.Linear(64, 16)
    # FCN para value
    self.linear_1 = nn.Linear(1536, 256)
    self.linear_2 = nn.Linear(256, 64)
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
    if len(x.shape) == 3:
      x = x.unsqueeze(0)
  # Forward común
    u1 = self.conv1(x)
    h1 = F.relu(u1)
    f1 = self.max_p1(h1)
    # print(f1.shape)
    u2 = self.conv2(f1)
    h2 = F.relu(u2)
    f2 = self.max_p2(h2)
    # print(f2.shape)
    u3 = self.conv3(f2)
    h3 = F.relu(u3)
    f3 = self.max_p3(h3)
    # print(f3.shape)
    m = torch.flatten(input = f3, start_dim=1)
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

net = CNN()
optimizer = AdamW(net.parameters(), lr = 1e-4)
env = FireGrid_V4(20, burn_value=10, n_sims=100)
states = []
state = env.reset()
states.append(state)
step_data = []
I = 1
gamma = 0.99
now = datetime.datetime.now()
for i in range(1):
  done = False
  state = env.reset()
  while not done:
      I = 1
      policy, value = net.forward(copy.copy(state))
      action = policy.multinomial(1)
      next_state, reward, done = env.step(action.detach())
      _, value_next_state = net.forward(copy.copy(next_state))
      I *= gamma
      step_data.append([action, reward, policy, value, value_next_state, I])
      state = next_state
data = DataLoader(step_data, len(step_data), shuffle=False, pin_memory=True)
# print(len(step_data))
for action_t, reward_t, policy_t, value_t, value_next_state_t, discounts in data:
    # now = datetime.datetime.now()
    target = reward_t + gamma * value_next_state_t
    net.zero_grad()
    critic_loss = F.mse_loss(value_t.squeeze(), target)
    # critic_loss.backward(retain_graph=True)
    # now = datetime.datetime.now()
    action_t_s = action_t.squeeze()
    policy_t_s = policy_t.squeeze()
    advantage = (target - value_t).squeeze()
    log_probs = torch.log(policy_t.squeeze() + 1e-6)
    action_log_probs = log_probs.gather(1, action_t.squeeze().unsqueeze(1))
    entropy = -torch.sum(policy_t.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
    actor_loss = torch.sum(- discounts * action_log_probs * advantage - 0.02*entropy)
    total_loss = critic_loss + actor_loss
    total_loss.backward()
    # later = datetime.datetime.now()
    optimizer.step()
later = datetime.datetime.now()
print(later-now)
# x, y = net.forward(state)
# print(x.shape)
# pytorch_total_params = sum(p.numel() for p in net.parameters())
# print(pytorch_total_params)