from enviroment.firegrid_v3 import FireGrid_V3
import torch
from torch import nn as nn
net = nn.Sequential(
    nn.Conv2d(in_channels=2, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(0,2),
    nn.Linear(80, 16),
    nn.Softmax(dim = -1)
)
env = FireGrid_V3(20)
state = env.reset()
print(state)
a = net(state)
print(a)
print(state.shape)
state, r, done = env.step(torch.tensor(15))
print(state.shape)
print(state)
print(r)
print(done)
a = net(state)
print(a)
act = a.multinomial(1).detach()
print(act)
log_probs_t = torch.log(a + 1e-6)
print(log_probs_t)
action_log_prob_t = log_probs_t.gather(-1, torch.tensor(15))
print(action_log_prob_t)