import numpy as np
import torch
import json

class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, combined=False, temporal = False, env = "FG", version = 1, size = 20, n_envs = 16):
        self.size = size
        self.version = version
        if temporal:
            self.buffer = ReplayMemoryTemporal(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, combined=combined, env=env, size=size, n_envs=n_envs)
        else:
            self.buffer = ReplayMemoryUnTemporal(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, combined=combined, n_envs=n_envs)

    def is_sufficient(self):
        return self.mem_cntr > self.buffer.batch_size

    def load_demonstrations(self, n):
        file = open(f"algorithms/dpv/demonstrations/Sub{self.size}x{self.size}_full_grid_{self.version}.json")
        demonstrations = json.load(file)
        n_dem = 0
        for i in demonstrations.keys():
            for j in demonstrations[i].keys():
                self.buffer.store_transition_indiv(np.array(demonstrations[i][j][0]), demonstrations[i][j][1], demonstrations[i][j][2],np.array(demonstrations[i][j][4]), int(j), demonstrations[i][j][3])
                n_dem+=1
                if n_dem == n:
                    break
class ReplayMemoryUnTemporal:
    def __init__(self, input_dims, max_mem, batch_size, n_envs, combined=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.combined = combined
        self.n_envs = n_envs
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)


    def store_transition(self, state, action, reward, state_, step, done):
        for i in range(self.n_envs):
            self.store_transition_indiv(state[i], action[i], reward[i], state_[i], step, done)
    
    def store_transition_indiv(self, state, action, reward, state_, step, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.mem_cntr += 1

    def sample_memory(self):
        offset = 1 if self.combined else 0
        max_mem = min(self.mem_cntr, self.mem_size) - offset
        batch = np.random.choice(max_mem, self.batch_size-offset,
                                 replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        if self.combined:
            index = self.mem_cntr % self.mem_size - 1
            last_action = self.action_memory[index]
            last_state = self.state_memory[index]
            last_new_state = self.new_state_memory[index]
            last_reward = self.reward_memory[index]

            actions = np.append(self.action_memory[batch], last_action)
            states = np.vstack((self.state_memory[batch], last_state))
            new_states = np.vstack((self.new_state_memory[batch],
                                   last_new_state))
            rewards = np.append(self.reward_memory[batch], last_reward)

        return torch.Tensor(states).to(self.device),  torch.Tensor(actions).to(self.device),  torch.Tensor(rewards).to(self.device),  torch.Tensor(new_states).to(self.device)

    


class ReplayMemoryTemporal:
    def __init__(self, input_dims, max_mem, batch_size, n_envs, combined=False, env = "FG", size = 20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if env == "FG":
            quant = int((size**2)*0.05)
            if quant%2 == 0:
                quant += 1
            self.ep_len = quant
        else:
            self.ep_len = (size//2)**2
        self.mem_size = max_mem
        self.batch_size = batch_size // self.ep_len
        self.mem_cntr = 0
        self.combined = combined
        self.n_envs = n_envs
        self.state_memory = np.zeros((self.mem_size, self.ep_len, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, self.ep_len, *input_dims),
                                         dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, self.ep_len), dtype=np.int32)
        self.reward_memory = np.zeros((self.mem_size, self.ep_len), dtype=np.float32)

    def store_transition(self, state, action, reward, state_, step, done):
        for i in range(self.n_envs):
            self.store_transition_indiv(state[i], action[i], reward[i], state_[i], step, done)

    def store_transition_indiv(self, state, action, reward, state_, step, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index][step] = state
        self.action_memory[index][step] = action
        self.reward_memory[index][step] = reward
        self.new_state_memory[index][step] = state_
        if done:
            self.mem_cntr += 1

    def sample_memory(self):
        offset = 1 if self.combined else 0
        max_mem = min(self.mem_cntr, self.mem_size) - offset
        batch = np.random.choice(max_mem, self.batch_size-offset,
                                 replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]

        if self.combined:
            index = self.mem_cntr % self.mem_size - 1
            last_action = self.action_memory[index]
            last_state = self.state_memory[index]
            last_new_state = self.new_state_memory[index]
            last_reward = self.reward_memory[index]

            actions = np.append(self.action_memory[batch], last_action)
            states = np.vstack((self.state_memory[batch], last_state))
            new_states = np.vstack((self.new_state_memory[batch],
                                   last_new_state))
            rewards = np.append(self.reward_memory[batch], last_reward)

        return torch.Tensor(states).flatten(0, 1).to(self.device),  torch.Tensor(actions).flatten(0, 1).to(self.device),  torch.Tensor(rewards).flatten(0, 1).to(self.device),  torch.Tensor(new_states).flatten(0, 1).to(self.device)
