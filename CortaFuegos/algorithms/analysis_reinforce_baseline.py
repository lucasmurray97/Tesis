import os
import torch
import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from reinforce_baseline import reinforce_baseline
from cnn_a2c import CNN
# We create the enviroment
env = FireGrid_V4(20, burn_value=10, n_sims=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN()
net.to(device)
episodes = 5000
version = "v4"
# Let's plot the output of the policy net before training
state = env.reset()
for i in range(100):
    _, a = net.forward(state)
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} before training")
    plt.savefig(f"figures/{version}/reinforce_baseline/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()

# Let's check the value for the initial state:
_, v = net.forward(start_state)
a = v.detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")

plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]
stats = reinforce_baseline(env, net, episodes, version, plot_episode, beta=0.1)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/{version}/reinforce_baseline/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/reinforce_baseline/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/reinforce_baseline/critic_loss.png")
plt.show() 

# Let's check the value for the initial state after training:
_, v = net.forward(start_state)
a = v.detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")

