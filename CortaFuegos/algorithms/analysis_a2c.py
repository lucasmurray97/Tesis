import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.parallel_firegrid import Parallel_Firegrid
from a2c import actor_critic
# We create the enviroment
env = Parallel_Firegrid(20, burn_value=20)
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

# We create the value network:
value_net= nn.Sequential(
    nn.Linear(space_dims, 512),
    nn.LeakyReLU(),
    nn.Linear(512, 128),
    nn.LeakyReLU(),
    nn.Linear(128, 64),
    nn.LeakyReLU(),
    nn.Linear(64, 1)
)

# Let's check the policy's output in the initial state
start_tensor = start_state[0]
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state")
plt.savefig("figures/a2c/initial_probs.png")
plt.show()

# Let's check the value for the initial state:
start_tensor = start_state[0]
a = value_net(start_tensor).detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")


stats = actor_critic(env, policy, value_net, 100)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(100), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title("Returns for 100 episodes")
plt.savefig("figures/a2c/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(100), stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 100 episodes")
plt.savefig("figures/a2c/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot(np.arange(100), stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 100 episodes")
plt.savefig("figures/a2c/critic_loss.png")
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
plt.savefig("figures/a2c/initial_probs_ater_training.png")
plt.show()


# Let's check the value for the initial state after training:
start_tensor = start_state[0]
a = value_net(start_tensor).detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")

# Let's reconstruct the trajectory obtained
state = env.reset()
mat = state[0].reshape(20,20).numpy()
figure5 = plt.figure()
plt.clf()
plt.matshow(mat)
plt.savefig("figures/a2c/trajectory/initial_state.png")
plt.show()
for i in range(100):
    mat = state[0].reshape(20,20).numpy()
    a = policy(state)
    selected = [a.argmax() for i in range(12)]
    torch.Tensor(selected)
    state, done, _ = env.step(selected)
    figure5 = plt.figure()
    plt.clf()
    plt.matshow(mat)
    plt.savefig("figures/a2c/trajectory/" + str(i) + ".png")
    plt.show()





