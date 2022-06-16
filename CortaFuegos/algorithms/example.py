from enviroment.firegrid import FireGrid
from enviroment.parallel_firegrid import Parallel_Firegrid
import time
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
env = FireGrid(20)
# start = time.time()
# for i in range(1):
#     env.reset()
#     for i in range(100):
#         state, r, done = env.step(env.random_action())
# print(state)
# end = time.time()
# print(f"Test took {end-start} secs")
env = Parallel_Firegrid(20)
state = env.reset()
print(state[0].shape)
state = state[0]
for i in range(400):
    state[i] = -1
print(state)
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
# net.apply(net._init_weigths)
a = net(state)
print(a)
print(net.layer)

# lin = nn.Linear(400, 2048)
# lk = nn.LeakyReLU()
# out1 = lin(state)
# out = lk(out1)
# print(out)


