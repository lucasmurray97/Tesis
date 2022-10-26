import sys
import torch
import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from enviroment.firegrid_v5 import FireGrid_V5
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
from algorithms.reinforce import reinforce
from nets.cnn_policy_v1 import CNN as CNN_2
from nets.cnn_policy_v1_2 import CNN as CNN_3
from utils.plot_progress import plot_moving_av
from enviroment.utils.parallel_wrapper import Parallel_Wrapper 
n_envs = 8
# We retrieve the arguments from standard input
_, size, env_version, net_version, episodess, window_, gamma_, alpha_, beta_, instance, test_ = sys.argv
size = int(size)
window = int(window_)
episodes = int(episodess)
gamma = float(gamma_)
alpha = float(alpha_)
beta = float(beta_)
test = bool(test_)
# We create the enviroment
if env_version == "v6":
    env = Parallel_Wrapper(FireGrid_V6, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 50})
elif env_version == "v7":
    env = Parallel_Wrapper(FireGrid_V7, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 5, "n_sims_final" : 50})
elif env_version == "v8":
    env = Parallel_Wrapper(FireGrid_V8, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 5, "n_sims_final" : 50})
else:
    raise("Non existent version of enviroment")
start_state = env.reset()[0]
space_dims = env.envs[0].get_space_dims()[0]**2
action_dims = env.envs[0].get_action_space_dims()
# We create net
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if net_version == "net_2_2":
    net = CNN_2(size)
elif net_version == "net_3_2":
    net = CNN_3(size)
else:
    raise("Non existent version of network")
net.to(device)
save = True
update = 1
# Let's plot the output of the policy net before training
if test:
    path = "figures_tuning"
else:
    path = "figures"
print("Plotting pre-probs")
state = env.reset()[0]
for i in range((env.size//2)**2):
    a = net.forward(state.unsqueeze(0))
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} before training")
    plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/reinforce/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.envs[0].sample_space()
print("Finished plotting pre-probs")

plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
stats = reinforce(env, net, episodes, env_version, net_version, plot_episode, alpha = alpha, gamma = gamma, beta = beta, instance = instance, test = test)

# Guardamos los parametros de la red
if save:
    path_ = f"./weights/{env_version}/{net_version}/ppo.pth"
    torch.save(net.state_dict(), path_)
############################# Plots #################################################################
plot_moving_av(stats["Returns"], episodes*n_envs, env_version, net_version, "reinforce", window = window, instance = instance, test = test)


figure3 = plt.figure()
plt.clf()
plt.plot([i for i in range(episodes)], stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/reinforce/loss.png")
plt.show() 



