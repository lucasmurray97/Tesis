from enviroment.firegrid import FireGrid
import torch
from torch import nn as nn
net = nn.Sequential(
    nn.Linear(400, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 16),
    nn.Softmax(dim = -1)
)
env = FireGrid(20)
state = env.reset()
a = net(state)
# print(a)
# print(state.shape)
# state, r, done = env.step(15)
# print(state.shape)
# print(r)
# print(done)
# a = net(state)
print(a)
act = a.multinomial(1).detach()
print(act)