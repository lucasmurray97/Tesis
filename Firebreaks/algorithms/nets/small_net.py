import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked
from nets.small_net_v1 import CNN_V1
from Firebreaks.algorithms.nets.small_net_v2 import CNN_V2_FG
# Red estilo pytorch
class CNN:
  def __init__(self, env, version, grid__size = 20, input_size = 2, output_size = 16, value = True):
    self.env = env
    self.version = version
    self.grid_size = grid__size
    self.input_size = input_size
    self.output_size = output_size
    self.value = value
    
  # Computa la pasada hacia adelante
  def forward(self, x, mask = None):
    if self.env == "full_grid" and self.version == "v2":
      return 
