import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked, generate_mask, Q_Mask
# Red estilo pytorch
class CNN_BIG_Q_v2(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True, forbidden = [], only_q = False, version = 1, gpu=False):
    super(CNN_BIG_Q_v2, self).__init__()
    if gpu:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        self.device = torch.device('cpu')
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    self.forbidden = forbidden
    self.mask = Q_Mask(self.forbidden, version,gpu)
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=16, kernel_size=(2,2), stride=1, padding = 2, bias = True)
    self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.max_p1 = nn.MaxPool2d(2, stride=2)
    self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.conv4 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.max_p2 = nn.MaxPool2d(2, stride=1)
    self.only_q = only_q
  

    # FCN para Advantage
    if self.grid_size == 6:
      self.linear1 = nn.Linear(32, 48)
    elif self.grid_size == 10:
        self.linear1 = nn.Linear(288, 48)
    elif grid__size == 20:
       self.linear1 = nn.Linear(2048, 48)
    elif grid__size == 40:
       self.linear1 = nn.Linear(10368, 48)
    else:
      raise("Non existent grid size")
    self.linear2 = nn.Linear(48, 32)
    self.linear3 = nn.Linear(32, self.output_size)
    
    # FCN para value
    if not self.only_q:
      if self.grid_size == 6:
        self.linear_1 = nn.Linear(32, 48)
      elif self.grid_size == 10:
          self.linear_1 = nn.Linear(288, 48)
      elif grid__size == 20:
        self.linear_1 = nn.Linear(2048, 48)
      elif grid__size == 40:
       self.linear1 = nn.Linear(10368, 48)
      else:
        raise("Non existent grid size")
      self.linear_2 = nn.Linear(48, 32)
      self.linear_3 = nn.Linear(32, 1)

    # Inicializamos los parametros de la red:
    nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv3.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.conv4.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear1.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear2.weight, mode='fan_in', nonlinearity='relu')
    nn.init.kaiming_uniform_(self.linear3.weight, mode='fan_in', nonlinearity='linear')
    
    if not self.only_q:
      nn.init.kaiming_uniform_(self.linear_1.weight, mode='fan_in', nonlinearity='relu')
      nn.init.kaiming_uniform_(self.linear_2.weight, mode='fan_in', nonlinearity='relu')
      nn.init.kaiming_uniform_(self.linear_3.weight, mode='fan_in', nonlinearity='linear')

  # Computa la pasada hacia adelante
  def forward(self, x):
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
    # Forward Advantage
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    h4 = F.relu(u4)
    u5 = self.linear3(h4)
    adv_pred = u5
    mask = self.mask.filter(x)
    if self.only_q:
      masked_q = torch.where(condition = mask, input = adv_pred, other = torch.finfo(adv_pred.dtype).min)
      return masked_q
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    u_4 = self.linear_2(h_3)
    h_4 = F.relu(u_4)
    value_pred = self.linear_3(h_4)
    q_pred = (value_pred + (adv_pred - adv_pred.mean(dim=1, keepdim=True)))
    masked_q = torch.where(condition = mask, input = q_pred, other = torch.finfo(q_pred.dtype).min)
    return masked_q, value_pred

  def sample(self, q, state):
    actions = torch.argmax(q, dim=1).unsqueeze(1)
    return actions.to(self.device)

  def max_indiv(self, q, state):
    filter = self.mask.filter_indiv(state)
    filtered_q = q[filter]
    max = torch.amax(filtered_q, dim=0)
    return max

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
