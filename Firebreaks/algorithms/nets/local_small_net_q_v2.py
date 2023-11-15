import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked, generate_mask, Q_Mask
from nets.localconv import LocallyConnected2d
# Red estilo pytorch
class LOCAL_CNN_SMALL_Q_v2(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True, forbidden = [], only_q = False, version = 1, gpu=False):
    super(LOCAL_CNN_SMALL_Q_v2, self).__init__()
    if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device('cpu')
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    self.forbidden = forbidden
    self.mask = Q_Mask(self.forbidden, version)
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    out_size_1 = (self.grid_size - 2)//1 + 1
    self.conv1 = LocallyConnected2d(in_channels=self.input_size, out_channels=32, kernel_size=2, stride=1, bias = True, output_size = (out_size_1,out_size_1))
    out_size_2 = (out_size_1 - 2)//2 + 1
    self.conv2 = LocallyConnected2d(in_channels=32, out_channels=32, kernel_size=2, stride=2, bias = True, output_size = (out_size_2, out_size_2))
    self.max_p1 = nn.MaxPool2d(2, stride=2)
    self.only_q = only_q
  

    # FCN para Advantage
    if self.grid_size == 4:
      self.linear1 = nn.Linear(256, 256)
    elif self.grid_size == 6:
        self.linear1 = nn.Linear(32, 128) 
    elif self.grid_size == 10:
      self.linear1 = nn.Linear(128, 128)
    elif grid__size == 20:
       self.linear1 = nn.Linear(512, 128)
    else:
      raise("Non existent grid size")
    self.linear2 = nn.Linear(128, self.output_size)
    
    # FCN para value
    if not self.only_q:
      if self.grid_size == 6:
          self.linear_1 = nn.Linear(32, 128) 
      elif self.grid_size == 10:
          self.linear_1 = nn.Linear(128, 128)
      elif grid__size == 20:
        self.linear_1 = nn.Linear(512, 128)
      else:
        raise("Non existent grid size")
      self.linear_2 = nn.Linear(128, 1)

    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='linear')
    if not self.only_q:
      nn.init.kaiming_uniform_(self.linear_1.weight, mode='fan_in', nonlinearity='relu')
      nn.init.kaiming_uniform_(self.linear_2.weight, mode='fan_in', nonlinearity='linear')

  # Computa la pasada hacia adelante
  def forward(self, x):
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
    m = torch.flatten(input = f1, start_dim=1)
    # print(m.shape)
    # Forward Advantage
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    adv_pred = u4
    mask = self.mask.filter(x)
    if self.only_q:
      masked_q = torch.where(condition = mask, input = adv_pred, other = torch.finfo(adv_pred.dtype).min)
      return masked_q
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    value_pred = self.linear_2(h_3)
    q_pred = (value_pred + (adv_pred - adv_pred.mean(dim=1, keepdim=True)))
    masked_q = torch.where(condition = mask, input = q_pred, other = torch.finfo(q_pred.dtype).min)
    return masked_q, value_pred

  def sample(self, q, state):
    actions = torch.argmax(q, dim=1).unsqueeze(1)
    return actions.to(self.device)

  def max(self, q, state):
    max_vals, max_idx = torch.max(q, dim=1)
    return max_vals

  def je_loss(self, action, q_target, state, dem, l = 0.1):
    loss = 0
    for i in range(state.shape[0]):
      if dem[i]:
        loss += self.je_loss_indiv(action[i], q_target[i], state[i], l)
    return loss

  def je_loss_indiv(self, action, q_target, state, l):
    return torch.amax(q_target + l, dim=0)