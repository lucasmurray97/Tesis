import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.firegrid_v4 import FireGrid_V4
import copy
import random
import torch.nn.functional as F
import torch.utils.data as data_utils
env = FireGrid_V4(20, burn_value=50)
start_state = env.reset()
space_dims = env.get_space_dims()[0]**2
action_dims = env.get_action_space_dims()

q_network = nn.Sequential(
    nn.Conv2d(in_channels=3, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Conv2d(in_channels=20, out_channels=20, kernel_size=(5,5)),
    nn.LeakyReLU(),
    nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),
    nn.Flatten(0,2),
    nn.Linear(80, 16)
)

target_q_network = copy.deepcopy(q_network).eval()

def policy(env, state, epsilon=0.):
    if torch.rand(1) < epsilon:
        return env.random_action()
    else:
        av = q_network(state).detach()
        max_a = torch.argmax(av, dim=-1, keepdim=True).to(torch.float)
        return max_a
# action = policy(env, start_state, epsilon=0.1)
# print(action)

class ReplayMemory:
    
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def insert(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        assert self.can_sample(batch_size)
        sample = random.sample(self.memory, batch_size)
        return sample

    def can_sample(self, batch_size):
        return len(self.memory) >= batch_size * 10

    def __len__(self):
        return len(self.memory)

# memory = ReplayMemory()
# action = env.random_action()
# next_state, reward, done = env.step(action)
# transition = memory.insert([start_state, action, reward, done, next_state])
# states = []
# for i in range(500):
#     state = env.sample_space()
#     states.append(state)
#     action = env.random_action()
#     next_state, reward, done = env.step(action)
#     states.append(next_state)
#     memory.insert([state, action, reward, torch.Tensor([done]), next_state])
# random_states = random.sample(states, 32)
# dataloader = data_utils.DataLoader(random_states, batch_size=32, shuffle=True)
# batch = next(iter(dataloader))
# print(batch.shape)
# batch = memory.sample(32)
# for i in batch:
#     val = q_network(i[0])
#     print(val.shape)
# print(memory.can_sample(32))
# batch = memory.sample(32)




def deep_q_learning(q_network, policy, episodes, alpha = 0.0001, batch_size = 32, gamma = 0.99, epsilon = 0.1):
    optim = AdamW(q_network.parameters(), lr = alpha)
    memory = ReplayMemory()
    stats = {"MSE Loss": [], "Returns": []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return = 0
        while not done:
            action = policy(env, state, epsilon)
            next_state, reward, done = env.step(action)
            done = torch.Tensor([done])
            memory.insert([state, action, reward, done, next_state])
            
            if memory.can_sample(batch_size):
                batch = memory.sample(batch_size)
                for i in batch:
                    state, action, reward, done, next_state = i
                    qsa_b = q_network(state).gather(-1, action.to(torch.int64))
                    next_qsa_b = target_q_network(next_state)
                    next_qsa_b = torch.max(next_qsa_b, dim = -1, keepdim = True)[0]
                    target_b = reward +  gamma*next_qsa_b
                    loss = F.mse_loss(qsa_b, target_b)
                    q_network.zero_grad()
                    loss.backward()
                    optim.step()        
            state = next_state
            ep_return += reward.item()
        stats["Returns"].append(ep_return)
        if memory.can_sample(batch_size):
            stats["MSE Loss"].append(loss)
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())
    return stats
stats = deep_q_learning(q_network, policy, 10)
print(len(stats["MSE Loss"]))