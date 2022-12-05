import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked
# Red estilo pytorch
class CNN(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True):
    super(CNN, self).__init__()
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=32, kernel_size=(1,1), stride=1, padding = 3, bias = True)
    self.max_p1 = nn.MaxPool2d(2, stride=1)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(2,2), stride=1, padding = 3, bias = True)
    self.max_p2 = nn.MaxPool2d(2, stride=2)
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5,5), padding = 1, bias = True)
    self.max_p3 = nn.MaxPool2d(2, stride=3)

    # FCN para policy
    if self.grid_size == 4:
      self.linear1 = nn.Linear(256, 256)
    elif self.grid_size == 8:
        self.linear1 = nn.Linear(256, 256)
    elif self.grid_size == 2:
      self.linear1 = nn.Linear(64, 256)
    else:
       self.linear1 = nn.Linear(1024, 256)
    self.linear2 = nn.Linear(256, 64)
    self.linear3 = nn.Linear(64, self.output_size)
    # FCN para value
    if self.grid_size == 4:
      self.linear_1 = nn.Linear(256, 256)
    elif self.grid_size == 8:
        self.linear_1 = nn.Linear(256, 256)
    elif self.grid_size == 2:
      self.linear_1 = nn.Linear(64, 256)
    else:
       self.linear_1 = nn.Linear(1024, 256)
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
  def forward(self, x, mask = None):
    if len(x.shape) == 3:
      x = x.unsqueeze(0)
    # print(x.shape)
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
    head_masked = CategoricalMasked(logits=u5, mask = mask)
    if not self.value:
        return head_masked.probs, head_masked.entropy()
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    u_4 = self.linear_2(h_3)
    h_4 = F.relu(u_4)
    value_pred = self.linear_3(h_4)
    entropy = head_masked.entropy()
    return head_masked.probs, value_pred, entropy
