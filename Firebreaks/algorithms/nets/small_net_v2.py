import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked
# Red estilo pytorch
class CNN_SMALL_V2(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True):
    super(CNN_SMALL_V2, self).__init__()
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=32, kernel_size=(2,2), stride=2, padding = 0, bias = True)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(4,4), stride=4, padding = 0, bias = True)
    self.max_p1 = nn.MaxPool2d(4, stride=4)
  

    # FCN para policy
    if self.grid_size == 4:
      self.linear1 = nn.Linear(256, 256)
    elif self.grid_size == 6:
        self.linear1 = nn.Linear(32, 128) 
    elif self.grid_size == 10:
      self.linear1 = nn.Linear(288, 128)
    elif grid__size == 20:
       self.linear1 = nn.Linear(4608, 128)
    else:
      raise("Non existent grid size")
    self.linear2 = nn.Linear(128, self.output_size)
    # FCN para value
    if self.grid_size == 6:
        self.linear_1 = nn.Linear(32, 128) 
    elif self.grid_size == 10:
        self.linear_1 = nn.Linear(288, 128)
    elif grid__size == 20:
       self.linear_1 = nn.Linear(4608, 128)
    else:
      raise("Non existent grid size")
    self.linear_2 = nn.Linear(128, 1)

    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear_1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear_2.weight, mode='fan_in', nonlinearity='relu')

  # Computa la pasada hacia adelante
  def forward(self, x, mask = None):
    if len(x.shape) == 3:
      x = x.unsqueeze(0)
    # print(x.shape)
  # Forward común
    u1 = self.conv1(x)
    h1 = F.relu(u1)
    # print(h1.shape)
    u2 = self.conv2(h1)
    h2 = F.relu(u2)
    # print(h2.shape)
    f1 = self.max_p1(h2)
    # print(f1.shape)
    m = torch.flatten(input = f1, start_dim=1)
    # print(m.shape)
    # Forward Policy
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    head_masked = CategoricalMasked(logits=u4, mask = mask)
    if not self.value:
        return head_masked.probs, head_masked.entropy()
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    value_pred = self.linear_2(h_3)
    entropy = head_masked.entropy()
    return head_masked.probs, value_pred, entropy
