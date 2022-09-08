import os
import torch
import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from a2c import actor_critic
from a2c_batched import actor_critic_batched
from cnn_a2c import CNN
# We create the enviroment
env = FireGrid_V4(20, burn_value=20, n_sims=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN()
net.to(device)
episodes = 50
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
    plt.savefig(f"figures/{version}/a2c_batched/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()

# Let's check the value for the initial state:
_, v = net.forward(start_state)
a = v.detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")

plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]
stats = actor_critic_batched(env, net, episodes, version, plot_episode, beta=0.1)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c_batched/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot([i for i in range(episodes + 1) if i%5 == 0 and i != 0], stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c_batched/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot([i for i in range(episodes + 1) if i%5 == 0 and i != 0], stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c_batched/critic_loss.png")
plt.show() 

# Let's check the value for the initial state after training:
_, v = net.forward(start_state)
a = v.detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")

