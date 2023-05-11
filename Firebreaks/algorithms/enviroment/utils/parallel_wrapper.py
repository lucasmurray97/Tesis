import threading
import torch
from torch import nn as nn
import numpy as np
import copy
class Parallel_Wrapper():
    def __init__(self, env, n_envs = 8, parameters = {}):
        self.size = parameters["size"]
        self.n_envs = n_envs
        self.env = env
        parameters["env_id"] = 0
        self.envs = [env(**parameters)] 
        self.device = torch.device('cpu')
        self.lock = threading.Lock()
        for i in range(self.n_envs - 1):
            parameters["env_id"] = i + 1
            self.envs.append(env(**parameters))
        self.env_shape = self.envs[0].shape
        self.forbidden_cells = self.envs[0].forbidden_cells

    def individual_reset(self, i, states):
            state = self.envs[i].reset()
            with self.lock:
                states[i] = state

    def reset(self):
            states = [0 for i in range(self.n_envs)]
            threads = []
            for i in range(self.n_envs):
                threads.append(threading.Thread(name = i, target = self.individual_reset(i, states)))
            for i in range(self.n_envs):
                threads[i].start()
            for i in range(self.n_envs):
                threads[i].join()
            return torch.stack(states).to(self.device)

    def individual_step(self, i, next_states, rewards, done_b, action):
        next_state, reward, done = self.envs[i].step(action)
        with self.lock:
            next_states[i] = next_state
            rewards[i] = torch.Tensor([reward])
            done_b[i] = done
    
    def step(self, action):
        next_states = [0 for i in range(self.n_envs)]
        rewards = [0 for i in range(self.n_envs)]
        done_b = [0 for i in range(self.n_envs)]
        threads = []
        for i in range(self.n_envs):
            threads.append(threading.Thread(name = i, target = self.individual_step(i, next_states, rewards, done_b, action[i])))
        for i in range(self.n_envs):
            threads[i].start()
        for i in range(self.n_envs):
            threads[i].join()
        return torch.stack(next_states).to(self.device), torch.stack(rewards).to(self.device), done_b[0]
    
    def random_action(self):
        actions = []
        for i in range(self.n_envs):
            actions.append(self.envs[i].random_action())
        return torch.stack(actions).to(self.device)
