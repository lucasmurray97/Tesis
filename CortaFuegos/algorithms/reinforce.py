import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid import FireGrid

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

# Let's check the policy's output in the initial state
start_tensor = torch.tensor(start_state.reshape(1, 400)).float()
a = policy(start_tensor)
f1 = plt.figure()
plt.bar(np.arange(16), a.detach().numpy().squeeze())
plt.xlabel("Actions") 
plt.ylabel("Action Probability") 
plt.title("Action probabilities in initial state")
plt.savefig("figures/reinforce/initial_probs.png")
plt.show()

# action = policy(start_tensor).multinomial(1).detach()
# print(action)

# Reinforce Algorithm:
def reinforce(policy, episodes, alpha = 0.001, gamma = 0.99):
    optim = AdamW(policy.parameters(), lr = alpha)
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        transitions = []
        ep_return = 0
        while not done:
            action = policy(torch.tensor(state.reshape(1, 400)).float()).multinomial(1).detach()
            next_state, reward, done = env.step(action.item())
            transitions.append([state, action, reward])
            ep_return += reward
            state = next_state
        G = torch.zeros((1, 1))
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma * G
            probs_t = policy(torch.tensor(state_t.reshape(1, 400)).float())
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)
            entropy_t = -torch.sum(probs_t * log_probs_t, dim = -1, keepdim = True)
            gamma_t = gamma**t
            pg_loss_t = -gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - 0.01*entropy_t).mean()
            policy.zero_grad()
            total_loss_t.backward()
            optim.step()
        stats["Loss"].append(total_loss_t.item())
        stats["Returns"].append(ep_return)
    return stats
stats = reinforce(policy, 500)


figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(500), stats["Returns"])
plt.xlabel("Episode") 
plt.ylabel("Returns") 
plt.title("Returns for 500 episodes")
plt.savefig("figures/reinforce/returns.png")
plt.show() 


figure3 = plt.figure()
plt.clf()
plt.plot(np.arange(500), stats["Loss"])
plt.xlabel("Episode") 
plt.ylabel("Loss") 
plt.title("Loss for 500 episodes")
plt.savefig("figures/reinforce/loss.png")
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
plt.savefig("figures/reinforce/initial_probs_ater_training.png")
plt.show()