from torchvision.models import mobilenet_v3_small, mobilenet_v3_large
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from nets.mask import CategoricalMasked, generate_mask, Q_Mask

class small_mobile(torch.nn.Module):
    def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True, forbidden = [], only_q = False, version = 1):
        super(small_mobile, self).__init__()
        self.device = torch.device('cpu')
        self.grid_size = grid__size
        self.input_size = input_size
        self.output_size = output_size
        self.value = value
        self.forbidden = forbidden
        self.mask = Q_Mask(self.forbidden, version)
        self.only_q = only_q
        self.mobile_net = mobilenet_v3_small(weight="DEFAULT")
        self.layers = []
        self.n_layers = 0
        self.freeze_from = 8
        for layer in self.mobile_net.features:
            self.layers.append(layer)
            if self.n_layers > self.freeze_from:
                for param in self.layers[self.n_layers].parameters():
                    param.requires_grad = False
            self.n_layers += 1
        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=3, kernel_size=1, stride=1, padding = 0, bias = True)
        self.conv_blocks = nn.Sequential(*self.layers)
        self.linear_1 = nn.Linear(576, 512)
        self.linear_2 = nn.Linear(512, self.output_size) 
        self.linear_1_2 = nn.Linear(576, 512)
        self.linear_2_2 = nn.Linear(512, 1) 
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        h = self.conv1(x)
        u = F.relu(h)
        h1 = self.conv_blocks(u)
        m = torch.flatten(input = h1, start_dim=1)
        u2 = self.linear_1(m)
        h2 = F.relu(u2)
        adv_pred = self.linear_2(h2)
        mask = self.mask.filter(x)
        if self.only_q:
            masked_q = torch.where(condition = mask, input = adv_pred, other = torch.finfo(adv_pred.dtype).min)
            return masked_q
        u3 = self.linear_1_2(m)
        h3 = F.relu(u3)
        value_pred = self.linear_2_2(h3)
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

class big_mobile(torch.nn.Module):
    def __init__(self, grid__size = 20, input_size = 2, output_size = 16, value = True, forbidden = [], only_q = False, version = 1):
        super(big_mobile, self).__init__()
        self.device = torch.device('cpu')
        self.grid_size = grid__size
        self.input_size = input_size
        self.output_size = output_size
        self.value = value
        self.forbidden = forbidden
        self.mask = Q_Mask(self.forbidden, version)
        self.only_q = only_q
        self.mobile_net = mobilenet_v3_large(weight="DEFAULT")
        self.layers = []
        self.n_layers = 0
        self.freeze_from = 8
        for layer in self.mobile_net.features:
            self.layers.append(layer)
            if self.n_layers > self.freeze_from:
                for param in self.layers[self.n_layers].parameters():
                    param.requires_grad = False
            self.n_layers += 1
        self.conv1 = nn.Conv2d(in_channels=self.input_size, out_channels=3, kernel_size=1, stride=1, padding = 0, bias = True)
        self.conv_blocks = nn.Sequential(*self.layers)
        self.linear_1 = nn.Linear(960, 512)
        self.linear_2 = nn.Linear(512, self.output_size) 
        self.linear_1_2 = nn.Linear(960, 512)
        self.linear_2_2 = nn.Linear(512, 1) 
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        h = self.conv1(x)
        u = F.relu(h)
        h1 = self.conv_blocks(u)
        m = torch.flatten(input = h1, start_dim=1)
        u2 = self.linear_1(m)
        h2 = F.relu(u2)
        adv_pred = self.linear_2(h2)
        mask = self.mask.filter(x)
        if self.only_q:
            masked_q = torch.where(condition = mask, input = adv_pred, other = torch.finfo(adv_pred.dtype).min)
            return masked_q
        u3 = self.linear_1_2(m)
        h3 = F.relu(u3)
        value_pred = self.linear_2_2(h3)
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

