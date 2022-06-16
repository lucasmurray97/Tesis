import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn, tanh
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid import FireGrid
from a2c import actor_critic
# We create the enviroment
env = FireGrid(20, burn_value=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

# We create the policy:
policy = nn.Sequential(
    nn.Linear(space_dims, 2048),
    nn.LeakyReLU(),
    nn.Linear(2048, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, action_dims[0]),
    nn.Softmax(dim = -1)
)

# We create the value network:
value_net= nn.Sequential(
    nn.Linear(space_dims, 2048),
    nn.LeakyReLU(),
    nn.Linear(2048, 1024),
    nn.LeakyReLU(),
    nn.Linear(1024, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 256),
    nn.LeakyReLU(),
    nn.Linear(256, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 1)
)

episodes = 1000
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
    plt.savefig(f"figures/a2c/{episodes}_ep/probabilities/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()

# Let's check the value for the initial state:
a = value_net(start_state).detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")


stats = actor_critic(env, policy, value_net, episodes, beta=0.1)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/a2c/{episodes}_ep/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/a2c/{episodes}_ep/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/a2c/{episodes}_ep/critic_loss.png")
plt.show() 

# Let's check the value for the initial state after training:
a = value_net(start_state).detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")

# Let's reconstruct the trajectory obtained
state = env.reset()
mat = state.reshape(20,20).numpy()
figure5 = plt.figure()
plt.clf()
plt.matshow(mat)
plt.savefig(f"figures/a2c/{episodes}_ep/trajectory/initial_state.png")
plt.show()
for i in range(100):
    mat = state.reshape(20,20).numpy()
    a = policy(state)
    f2 = plt.figure()
    plt.clf()
    plt.bar(np.arange(16), a.detach().numpy().squeeze())
    plt.xlabel("Actions") 
    plt.ylabel("Action Probability") 
    plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
    plt.savefig(f"figures/a2c/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
    plt.show()
    selected = a.multinomial(1).detach()
    state, done, _ = env.step(selected)
    figure5 = plt.figure()
    plt.clf()
    plt.matshow(mat)
    plt.savefig(f"figures/a2c/{episodes}_ep/trajectory/" + str(i) + ".png")
    plt.show()





