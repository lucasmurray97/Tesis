import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn, tanh
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from reinforce import reinforce
# We create the enviroment
env = FireGrid_V3(20, burn_value=10)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

# We create the policy:
policy = nn.Sequential(
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

episodes = 30000
version = "v3"
# Let's plot the output of the policy net before training
state = env.reset()
for i in range(100):
    a = policy(state)
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} before training")
    plt.savefig(f"figures/{version}/reinforce/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()
plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000]
stats = reinforce(env, policy, episodes, version, plot_episode)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/{version}/reinforce/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/reinforce/loss.png")
plt.show() 



