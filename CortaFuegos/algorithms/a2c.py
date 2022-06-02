import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid import FireGrid
import torch.nn.functional as F

# We create the enviroment:

env = FireGrid(20)
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
start_tensor = torch.tensor(start_state.reshape(1, 400)).float()
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state")
plt.savefig("figures/a2c/initial_probs.png")
plt.show()

# Let's check the value for the initial state:
start_tensor = torch.tensor(start_state.reshape(1, 400)).float()
a = value_net(start_tensor).detach().numpy().squeeze()
print(f"Value of initial state before training: {a}")


# A2C algorithm:
def actor_critic(policy, value_net, episodes, alpha = 1e-4, gamma = 0.99):
    policy_optim = AdamW(policy.parameters(), lr = alpha)
    value_net_optim = AdamW(value_net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            action = policy(torch.tensor(state.reshape(1, 400)).float()).multinomial(1).detach()
            next_state, reward, done = env.step(action.item())
            value = value_net(torch.tensor(state.reshape(1, 400)).float())
            target = reward + gamma * value_net(torch.tensor(next_state.reshape(1, 400)).float()).detach()
            critic_loss = F.mse_loss(value, target)
            value_net.zero_grad()
            critic_loss.backward()
            value_net_optim.step()
            
            advantage = (target - value).detach()
            probs = policy(torch.tensor(state.reshape(1, 400)).float())
            log_probs = torch.log(probs + 1e-6)
            action_log_probs = log_probs.gather(1, action)
            entropy = -torch.sum(probs * log_probs, dim = -1, keepdim = True)
            actor_loss = -I * action_log_probs * advantage - 0.01*entropy
            actor_loss = actor_loss.mean()
            policy.zero_grad()
            actor_loss.backward()
            policy_optim.step()
            
            ep_return += reward
            state = next_state
            I *= gamma
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return)
    return stats

stats = actor_critic(policy, value_net, 5000)

figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(5000), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title("Returns for 5000 episodes")
plt.savefig("figures/a2c/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(5000), stats["Actor Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 5000 episodes")
plt.savefig("figures/a2c/actor_loss.png")
plt.show() 

figure4 = plt.figure()
plt.clf()
plt.plot(np.arange(5000), stats["Critic Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 5000 episodes")
plt.savefig("figures/a2c/critic_loss.png")
plt.show() 




# Let's check the policy's output in the initial state after training
start_state = env.reset()
start_tensor = torch.tensor(start_state.reshape(1, 400)).float()
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state after training")
plt.savefig("figures/a2c/initial_probs_ater_training.png")
plt.show()


# Let's check the value for the initial state after training:
start_tensor = torch.tensor(start_state.reshape(1, 400)).float()
a = value_net(start_tensor).detach().numpy().squeeze()
print(f"Value of initial state after training: {a}")


