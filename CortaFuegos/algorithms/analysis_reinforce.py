import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.parallel_firegrid import Parallel_Firegrid
from reinforce import reinforce
# We create the enviroment
env = Parallel_Firegrid(20)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

# We create the policy:
policy = nn.Sequential(
    nn.Linear(space_dims, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, action_dims[0]),
    nn.Softmax(dim = -1)
)

# Let's check the policy's output in the initial state
start_tensor = start_state[0]
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state")
plt.savefig("figures/reinforce/initial_probs.png")
plt.show()


stats = reinforce(env, policy, 10)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(10), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title("Returns for 500 episodes")
plt.savefig("figures/reinforce/returns.png")
plt.show() 

figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(10), stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 500 episodes")
plt.savefig("figures/reinforce/loss.png")
plt.show() 

# Let's check the policy's output in the initial state after training
start_state = env.reset()
start_tensor = start_state[0]
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state after training")
plt.savefig("figures/reinforce/initial_probs_ater_training.png")
plt.show()

# Let's reconstruct the trajectory obtained
state = env.reset()
mat = state[0].reshape(20,20).numpy()
plt.clf()
plt.matshow(mat)
plt.savefig("figures/reinforce/trajectory/initial_state.png")
plt.show()
for i in range(100):
    mat = state[0].reshape(20,20).numpy()
    figure2 = plt.figure()
    plt.clf()
    plt.matshow(mat)
    plt.savefig("figures/reinforce/trajectory/" + str(i) + ".png")
    plt.show()
    a = policy(state)
    selected = [a.argmax() for i in range(12)]
    torch.Tensor(selected)
    state, _, _ = env.step(selected)





