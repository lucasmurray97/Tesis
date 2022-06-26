import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn, tanh
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid_v3 import FireGrid_V3
from reinforce import reinforce
# We create the enviroment
env = FireGrid_V3(20, burn_value=50)
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

episodes = 10000
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
    plt.savefig(f"figures/reinforce/{episodes}_ep/probabilities/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()

stats = reinforce(env, policy, episodes)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/reinforce/{episodes}_ep/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/reinforce/{episodes}_ep/loss.png")
plt.show() 

# Let's reconstruct the trajectory obtained
state = env.reset()
mat = state[0].reshape(20,20).numpy()
figure5 = plt.figure()
plt.clf()
plt.matshow(mat)
plt.savefig(f"figures/reinforce/{episodes}_ep/trajectory/initial_state.png")
plt.show()
for i in range(100):
    mat = state[0].reshape(20,20).numpy()
    a = policy(state)
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
    plt.savefig(f"figures/reinforce/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
    plt.show()
    selected = a.multinomial(1).detach()
    state, done, _ = env.step(selected)
    figure5 = plt.figure()
    plt.clf()
    plt.imshow(mat)
    plt.colorbar()
    plt.title(f"State {i} in agent's trajectory")
    plt.savefig(f"figures/reinforce/{episodes}_ep/trajectory/" + str(i) + ".png")
    plt.show()





