import threading
from enviroment.firegrid_v2 import FireGrid_V2
import torch
from torch import nn as nn
import numpy as np
n_cores = 12

class Parallel_Firegrid():
    def __init__(self, size, agent_id = -1, agent_dim = 2, burn_value = 10):
        self.size = size
        self.envs = [] 
        self.lock = threading.Lock()
        for i in range(n_cores):
            env = FireGrid_V2(size, agent_id, agent_dim, burn_value)
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
        return torch.Tensor(states).reshape((12, self.size**2))

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
            threads.append(threading.Thread(name = i, target = self.individual_step(i, next_states, rewards, done_b, action[i].item())))
        for i in range(n_cores):
            threads[i].start()
        for i in range(n_cores):
            threads[i].join()
        a = torch.Tensor(12, self.size**2)
        torch.cat(next_states, out=a)
        next_states = torch.reshape(a, (12, self.size**2))
        b = torch.Tensor(12, 1)
        torch.cat(rewards, out=b)
        rewards = torch.reshape(b, (12, 1))
        return next_states, rewards, done_b[0]
    
    def individual_step(self, i, next_states, rewards, done_b, action):
        next_state, reward, done = self.envs[i].step(action)
        with self.lock:
            next_states[i] = torch.Tensor(next_state).reshape(-1)
            rewards[i] = torch.Tensor([reward])
            done_b[i] = done
    def get_space_dims(self):
        return self.envs[0].get_space_dims()
    def get_action_space_dims(self):
        return self.envs[0].get_action_space_dims() 

    def show_state(self):
        for i in range(n_cores):
            print(f"Agente {i}:")
            self.envs[i].show_state()

    def sample_space(self):
        sample = []
        for i in range(n_cores):
            sample.append(torch.Tensor(self.envs[i].sample_space()).reshape(-1))
        a = torch.Tensor(12, self.size**2)
        torch.cat(sample, out=a)
        return torch.reshape(a, (12, self.size**2))
        



