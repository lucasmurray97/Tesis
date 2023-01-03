import numpy as np
import torch
import json
import pickle
class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, demonstrate = False, n_dem = None, combined=False, temporal = False, env = "FG", version = 1, size = 20, n_envs = 16, gamma = 1., landa = 1.):
        self.size = size if version == '1' else int(np.sqrt(size))
        self.version = version
        self.demonstrate = demonstrate
        self.n_dem = n_dem
        self.gamma = gamma
        self.landa = landa
        if temporal:
            self.buffer = ReplayMemoryTemporal(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, combined=combined, env=env, size=size, n_envs=n_envs)
        else:
            self.buffer = ReplayMemoryUnTemporal(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, combined=combined, n_envs=n_envs)
        if self.demonstrate:
            self.load_demonstrations()

    def is_sufficient(self):
        return self.buffer.mem_cntr > self.buffer.batch_size

    def load_demonstrations(self):
        print("Loading demonstrations!")
        file = open(f"algorithms/dpv/demonstrations/Sub{self.size}x{self.size}_full_grid_{self.version}.pkl", 'rb')
        demonstrations = pickle.load(file)
        n = 0
        for i in demonstrations.keys():
            D = 1.
            I = 1.
            for j in demonstrations[i].keys():
                self.buffer.store_transition_indiv(np.array(demonstrations[i][j][0]), demonstrations[i][j][1], demonstrations[i][j][2],np.array(demonstrations[i][j][4]), int(j), demonstrations[i][j][3], D, I)
                self.n_dem+=1
                D *= self.gamma
                I *= self.landa
                if self.n_dem == n:
                    break
                n+=1
        print(f"Succesfully loaded {n} demonstrations!")

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
        self.gammas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.landas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.dones_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, step, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_transition_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), step, done, gamma, landa)
    
    def store_transition_indiv(self, state, action, reward, state_, step, done, gamma, landa):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.gammas_memory[index] = gamma
        self.landas_memory[index] = landa
        self.dones_memory[index] = done

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
        gammas = self.gammas_memory[batch]
        landas = self.landas_memory[batch]
        dones = self.dones_memory[batch]

        if self.combined:
            index = self.mem_cntr % self.mem_size - 1
            last_action = self.action_memory[index]
            last_state = self.state_memory[index]
            last_new_state = self.new_state_memory[index]
            last_reward = self.reward_memory[index]
            last_gamma = self.gammas_memory[index]
            last_landa = self.landas_memory[index]

            actions = np.append(self.action_memory[batch], last_action)
            states = np.vstack((self.state_memory[batch], last_state))
            new_states = np.vstack((self.new_state_memory[batch],
                                   last_new_state))
            rewards = np.append(self.reward_memory[batch], last_reward)
            gammas = np.append(self.gammas_memory[batch], last_gamma)
            landas = np.append(self.landas_memory[batch], last_landa)

        return torch.Tensor(states).to(self.device),  torch.Tensor(actions).to(self.device),  torch.Tensor(rewards).to(self.device),  torch.Tensor(new_states).to(self.device), torch.Tensor(gammas).to(self.device), torch.Tensor(landas).to(self.device), torch.Tensor(dones).bool().to(self.device)

    


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
        self.gammas_memory = np.zeros((self.mem_size, self.ep_len), dtype=np.float32)
        self.landas_memory = np.zeros((self.mem_size, self.ep_len), dtype=np.float32)
        self.dones_memory = np.zeros((self.mem_size, self.ep_len), dtype=np.bool)



    def store_transition(self, state, action, reward, state_, step, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_transition_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), step, done, gamma, landa)
        if done:
            self.mem_cntr += 1
    def store_transition_indiv(self, state, action, reward, state_, step, done, gamma, landa):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index][step] = state
        self.action_memory[index][step] = action
        self.reward_memory[index][step] = reward
        self.new_state_memory[index][step] = state_
        self.gammas_memory[index][step] = gamma
        self.landas_memory[index][step] = landa
        self.dones_memory[index][step] = done
    def sample_memory(self):
        offset = 1 if self.combined else 0
        max_mem = min(self.mem_cntr, self.mem_size) - offset
        batch = np.random.choice(max_mem, self.batch_size-offset,
                                 replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        gammas = self.gammas_memory[batch]
        landas = self.landas_memory[batch]
        dones = self.dones_memory[batch]
        if self.combined:
            index = self.mem_cntr % self.mem_size - 1
            last_action = self.action_memory[index]
            last_state = self.state_memory[index]
            last_new_state = self.new_state_memory[index]
            last_reward = self.reward_memory[index]
            last_gamma = self.gammas_memory[index]
            last_landa = self.landas_memory[index]

            actions = np.append(self.action_memory[batch], last_action)
            states = np.vstack((self.state_memory[batch], last_state))
            new_states = np.vstack((self.new_state_memory[batch],
                                   last_new_state))
            rewards = np.append(self.reward_memory[batch], last_reward)
            gammas = np.append(self.gammas_memory[batch], last_gamma)
            landas = np.append(self.landas_memory[batch], last_landa)

        return torch.Tensor(states).flatten(0, 1).to(self.device),  torch.Tensor(actions).flatten(0, 1).to(self.device),  torch.Tensor(rewards).flatten(0, 1).to(self.device),  torch.Tensor(new_states).flatten(0, 1).to(self.device), torch.Tensor(gammas).flatten(0, 1).to(self.device), torch.Tensor(landas).flatten(0, 1).to(self.device), torch.Tensor(dones).flatten(0, 1).bool().to(self.device)
