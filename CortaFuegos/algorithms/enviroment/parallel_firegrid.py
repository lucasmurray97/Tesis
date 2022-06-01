import threading
from firegrid import FireGrid
import torch
from torch import nn as nn
import numpy as np
n_cores = 12

class Parallel_Firegrid():
    def __init__(self, size, agent_id = -1, agent_dim = 2):
        self.envs = [] 
        self.lock = threading.Lock()
        for i in range(n_cores):
            env = FireGrid(size, agent_id, agent_dim)
            self.envs.append(env)
    
    def reset(self):
        states = []
        threads = []
        for i in range(n_cores):
            states.append(0)
            threads.append(threading.Thread(name = i, target = self.individual_reset(i, states)))
        for i in range(n_cores):
            threads[i].start()
        for i in range(n_cores):
            threads[i].join()
        return torch.Tensor(states).reshape((20, 20, 12))

    def individual_reset(self, i, states):
        state = self.envs[i].reset()
        with self.lock:
            states[i] = state

    def step(self, action):
        next_states = []
        rewards = []
        done_b = []
        threads = []
        for i in range(n_cores):
            next_states.append(0)
            rewards.append(0)
            done_b.append(0)
            threads.append(threading.Thread(name = i, target = self.individual_step(i, next_states, rewards, done_b, action[i])))
        for i in range(n_cores):
            threads[i].start()
        for i in range(n_cores):
            threads[i].join()
        return torch.Tensor(next_states), torch.Tensor(rewards), done_b
    
    def individual_step(self, i, next_states, rewards, done_b, action):
        next_state, reward, done = self.envs[i].step(action)
        with self.lock:
            next_states[i] = next_state
            rewards[i] = reward
            done_b[i] = done

env = Parallel_Firegrid(20)
a = env.reset()
action = [i for i in range(12)]
next_state, reward, done = env.step(action)
print(next_state)
print(reward)
print(done)
# space_dims = 400
# action_dims = 16
# policy = nn.Sequential(
#     nn.Linear(space_dims, 512),
#     nn.LeakyReLU(),
#     nn.Linear(512, 128),
#     nn.LeakyReLU(),
#     nn.Linear(128, 64),
#     nn.LeakyReLU(),
#     nn.Linear(64, action_dims),
#     nn.Softmax(dim = -1)
# )
# action = policy(a).multinomial(1).detach()
# print(action)


