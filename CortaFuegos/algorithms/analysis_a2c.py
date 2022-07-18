import os
import torch
import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
from enviroment.firegrid import FireGrid
from enviroment.firegrid_v3 import FireGrid_V3
from enviroment.firegrid_v4 import FireGrid_V4
from a2c import actor_critic
# We create the enviroment
env = FireGrid_V4(20, burn_value=10, n_sims=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# We create the policy:
policy = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(0,2),
    nn.Linear(80, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 16),
    nn.Softmax(dim = -1)
)

policy.to(device)
# We create the value network:
value_net= nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(0,2),
    nn.Linear(80, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 32),
    nn.LeakyReLU(),
    nn.Linear(32, 1)
)
value_net.to(device)
episodes = 5000
version = "v4"
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
    plt.savefig(f"figures/{version}/a2c/pre_train/probs_before_training_"+ str(i) +".png")
    plt.show()
    state = env.sample_space()

# Let's check the value for the initial state:
a = value_net(start_state).detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")

plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000]
stats = actor_critic(env, policy, value_net, episodes, version, plot_episode, beta=0.1)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title(f"Returns for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot(np.arange(episodes), stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title(f"Loss for {episodes} episodes")
plt.savefig(f"figures/{version}/a2c/critic_loss.png")
plt.show() 

# Let's check the value for the initial state after training:
a = value_net(start_state).detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")

# Let's reconstruct the trajectory obtained
# state = env.reset()
# mat = state[0].reshape(20,20).numpy()
# figure5 = plt.figure()
# plt.clf()
# plt.matshow(mat)
# plt.savefig(f"figures/{version}/a2c/{episodes}_ep/trajectory/initial_state.png")
# plt.show()
# for i in range(100):
#     mat = state[0].reshape(20,20).numpy()
#     a = policy(state)
#     f2 = plt.figure()
#     plt.clf()
#     plt.bar(np.arange(16), a.detach().numpy().squeeze())
#     plt.xlabel("Actions") 
#     plt.ylabel("Action Probability") 
#     plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
#     plt.savefig(f"figures/{version}/a2c/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
#     plt.show()
#     selected = a.multinomial(1).detach()
#     state, done, _ = env.step(selected)
#     figure5 = plt.figure()
#     plt.imshow(mat)
#     plt.colorbar()
#     plt.title(f"State {i} in agent's trajectory")
#     plt.savefig(f"figures/{version}/a2c/{episodes}_ep/trajectory/" + str(i) + ".png")
#     plt.show()





