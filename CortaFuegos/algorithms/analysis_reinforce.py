import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn, tanh
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from enviroment.firegrid_v5 import FireGrid_V5
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
from cnn_policy_v1_2 import CNN as CNN_1
from utils.plot_progress import plot_moving_av
from reinforce import reinforce
# We create the enviroment
env = FireGrid_V6(20, burn_value=10, n_sims=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

# We create the policy:
net = CNN_1()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net.to(device)
episodes = 10
env_version = "v6"
net_version = "net_2_2"
save = True
# Let's plot the output of the policy net before training
state = env.reset()
for i in range(100):
    a = net.forward(state)
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} before training")
    plt.savefig(f"figures/{env_version}/{net_version}/reinforce/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()
plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000]
stats = reinforce(env, net, episodes, env_version, net_version, plot_episode)

# Guardamos los parametros de la red
if save:
    path = f"./weights/{env_version}/{net_version}/reinforce.pth"
    torch.save(net.state_dict(), path)

plot_moving_av(stats["Returns"], episodes, env_version, net_version, "reinforce", window = 10)


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{env_version}/{net_version}/reinforce/loss.png")
plt.show() 



