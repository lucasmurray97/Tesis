import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked
# Red estilo pytorch
class CNN_BIG_V1(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True):
    super(CNN_BIG_V1, self).__init__()
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=16, kernel_size=(2,2), stride=1, padding = 2, bias = True)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.max_p1 = nn.MaxPool2d(2, stride=2)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.max_p2 = nn.MaxPool2d(2, stride=1)

    # FCN para policy
    if self.grid_size == 6:
      self.linear1 = nn.Linear(32, 48)
    elif self.grid_size == 10:
        self.linear1 = nn.Linear(288, 48)
    elif grid__size == 20:
       self.linear1 = nn.Linear(2048, 48)
    self.linear2 = nn.Linear(48, 32)
    self.linear3 = nn.Linear(32, self.output_size)
    # FCN para value
    if self.grid_size == 6:
      self.linear_1 = nn.Linear(32, 48)
    elif self.grid_size == 10:
        self.linear_1 = nn.Linear(288, 48)
    elif grid__size == 20:
       self.linear_1 = nn.Linear(2048, 48)
    self.linear_2 = nn.Linear(48, 32)
    self.linear_3 = nn.Linear(32, 1)
    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
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
  # Forward común
    u1 = self.conv1(x)
    h1 = F.relu(u1)
    # print(h1.shape)
    u2 = self.conv2(h1)
    h2 = F.relu(u2)
    # print(h2.shape)
    f1 = self.max_p1(h2)
    # print(f1.shape)
    u3 = self.conv3(f1)
    h3 = F.relu(u3)
    # print(h3.shape)
    u4 = self.conv4(h3)
    h3 = F.relu(u4)
    # print(h3.shape)
    f2 = self.max_p2(h3)
    # print(f2.shape)
    m = torch.flatten(input = f2, start_dim=1)
    # print(m.shape)
    # Forward Policy
    u5 = self.linear1(m)
    h5 = F.relu(u5)
    u6 = self.linear2(h5)
    h6 = F.relu(u6)
    u7 = self.linear3(h6)
    head_masked = CategoricalMasked(logits=u7, mask = mask)
    if not self.value:
        return head_masked.probs, head_masked.entropy()
    # Forward value
    u_5 = self.linear_1(m)
    h_5 = F.relu(u_5)
    u_6 = self.linear_2(h_5)
    h_6 = F.relu(u_6)
    value_pred = self.linear_3(h_6)
    entropy = head_masked.entropy()
    return head_masked.probs, value_pred, entropy
