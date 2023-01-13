import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked, generate_mask, Q_Mask
# Red estilo pytorch
class CNN_SMALL_Q(torch.nn.Module):
  def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True, forbidden = [], only_q = False, version = 1):
    super(CNN_SMALL_Q, self).__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    self.forbidden = forbidden
    self.mask = Q_Mask(self.forbidden, version)
    # Definimos capas (automáticamente se registran como parametros)
    # Capas compartidas
    self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=32, kernel_size=(2,2), stride=1, padding = 0, bias = True)
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(2,2), stride=2, padding = 0, bias = True)
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
    # Forward Advantage
    u3 = self.linear1(m)
    h3 = F.relu(u3)
    u4 = self.linear2(h3)
    adv_pred = u4
    if self.only_q:
      return adv_pred
    # Forward value
    u_3 = self.linear_1(m)
    h_3 = F.relu(u_3)
    value_pred = self.linear_2(h_3)
    return adv_pred, value_pred

  def sample_indiv(self, q, state):
    filter = self.mask.filter_indiv(state)
    mask = dict(enumerate(filter.tolist(), 0))
    filtered_index = {}
    index = 0
    for i in range(len(mask)):
      if mask[i]:
        filtered_index[index] = i
        index+=1
    filtered_q = q[filter]
    action = torch.argmax(filtered_q, dim=0)
    return torch.Tensor([filtered_index[int(action.item())]])

  def sample(self, q, state):
    actions = []
    for i in range(state.shape[0]):
      actions.append(self.sample_indiv(q[i], state[i]))
    return torch.stack(actions).to(self.device)

  def max_indiv(self, q, state):
    filter = self.mask.filter_indiv(state)
    filtered_q = q[filter]
    max = torch.amax(filtered_q, dim=0)
    return max
  def max(self, q, state):
    maxs = []
    for i in range(state.shape[0]):
      maxs.append(self.max_indiv(q[i], state[i]))
    return torch.stack(maxs).to(self.device)
  def je_loss(self, action, q_target, state, dem, l = 0.1):
    loss = 0
    for i in range(state.shape[0]):
      if dem[i]:
        loss += self.je_loss_indiv(action[i], q_target[i], state[i], l)
    return loss

  def je_loss_indiv(self, action, q_target, state, l):
    filter = self.mask.filter_indiv(state)
    mask = dict(enumerate(filter.tolist(), 0))
    filtered_index = {}
    index = 0
    for i in range(len(mask)):
      if mask[i]:
        filtered_index[index] = i
        index+=1
    reversed_filtered_index = {int(value):int(key) for key, value in filtered_index.items()}
    filtered_q = q_target[filter]
    filtered_a = torch.Tensor([reversed_filtered_index[int(action.item())]])
    la = torch.Tensor([0 if i != filtered_a else l for i in range(filtered_q.shape[0])]).to(self.device)
    return torch.amax(filtered_q + la, dim=0)