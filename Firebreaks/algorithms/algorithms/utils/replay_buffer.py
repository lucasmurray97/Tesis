import numpy as np
import torch
import json
import pickle
class ReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, demonstrate = False, n_dem = None, prioritized = False, env = "FG", version = 1, size = 20, n_envs = 16, gamma = 1., landa = 1.):
        self.size = size
        self.version = version
        self.demonstrate = demonstrate
        self.n_dem = n_dem
        self.gamma = gamma
        self.landa = landa
        self.prioritized = prioritized
        if prioritized:
            self.buffer = PrioritizedReplayMemory(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, env=env, size=size, n_envs=n_envs)
        else:
            self.buffer = ReplayMemoryBaseline(input_dims=input_dims, max_mem=max_mem, batch_size=batch_size, n_envs=n_envs)
        if self.demonstrate:
            self.load_demonstrations()
        

    def is_sufficient(self):
        return self.buffer.mem_cntr > self.buffer.batch_size

    def load_demonstrations(self):
        print("Loading demonstrations!")
        file = open(f"algorithms/dpv/demonstrations/Sub{self.size}x{self.size}_full_grid_{self.version}.pkl", 'rb')
        demonstrations = pickle.load(file)
        n = 0
        end_flag = False
        for i in demonstrations.keys():
            D = 1.
            I = 1.
            for j in demonstrations[i].keys():
                self.buffer.store_dem_indiv(np.array(demonstrations[i][j][0]), demonstrations[i][j][1], demonstrations[i][j][2],np.array(demonstrations[i][j][4]), demonstrations[i][j][3], D, I)
                D *= self.gamma
                I *= self.landa
                if self.n_dem - 1 == n:
                    end_flag = True
                    break
                n+=1
            if end_flag:
                break
        print(f"Succesfully loaded {n + 1} demonstrations!")

class ReplayMemoryBaseline:
    def __init__(self, input_dims, max_mem, batch_size, n_envs):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.n_envs = n_envs
        self.dem_pivot = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)                                  
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.gammas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.landas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.dones_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_transition_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), done, gamma, landa)
    
    def store_transition_indiv(self, state, action, reward, state_, done, gamma, landa):
        index = (self.mem_cntr ) % (self.mem_size - self.dem_pivot) 
        self.state_memory[index + self.dem_pivot] = state
        self.action_memory[index + self.dem_pivot] = action
        self.reward_memory[index + self.dem_pivot] = reward
        self.new_state_memory[index + self.dem_pivot] = state_
        self.gammas_memory[index + self.dem_pivot] = gamma
        self.landas_memory[index + self.dem_pivot] = landa
        self.dones_memory[index + self.dem_pivot] = done
        self.mem_cntr += 1

    def store_dem(self, state, action, reward, state_, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_dem_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), done, gamma, landa)
    
    def store_dem_indiv(self, state, action, reward, state_, done, gamma, landa):
        index = self.dem_pivot % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.gammas_memory[index] = gamma
        self.landas_memory[index] = landa
        self.dones_memory[index] = done
        self.dem_pivot += 1


    def sample_memory(self):
        max_mem = min(self.mem_cntr + self.dem_pivot, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size,
                                 replace=False)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        gammas = self.gammas_memory[batch]
        landas = self.landas_memory[batch]
        dones = self.dones_memory[batch]
        dem = batch < self.dem_pivot
        return batch, torch.Tensor(states).to(self.device),  torch.Tensor(actions).to(self.device),  torch.Tensor(rewards).to(self.device),  torch.Tensor(new_states).to(self.device), torch.Tensor(gammas).to(self.device), torch.Tensor(landas).to(self.device), torch.Tensor(dones).bool().to(self.device), None, torch.Tensor(dem).to(self.device)

    def get_n_steps(self, indices):
        dem = indices < self.dem_pivot
        rewards = []
        next_states = []
        use_next = []
        for i in range(len(indices)):
            r, next_s, use_next_ = self.get_n_steps_indiv(indices[i], dem[i])
            rewards.append(r)
            next_states.append(next_s)
            use_next.append(use_next_)
        return torch.stack(rewards).to(self.device), torch.stack(next_states).to(self.device), torch.stack(use_next).to(self.device)
    
    def get_n_steps_indiv(self, indices, dem):
        if dem:
            start = 0
            end = self.dem_pivot - 1
        else:
            start = self.dem_pivot
            end = min(self.mem_cntr + self.dem_pivot, self.mem_size) - 1
        rewards = list(self.reward_memory[indices:min(indices + 2, end)])
        dones = list(self.dones_memory[indices:min(indices + 2, end)])
        if len(rewards) < 2:
            rewards = rewards + list(self.reward_memory[start:min(start + (2 - (len(rewards))), end)])
            dones = dones + list(self.dones_memory[start:min(start + (2 - (len(rewards))), end)])
        first_done = 0
        for i in dones:
            if i:
                break
            first_done += 1
        use_next = True
        n_rewards = rewards[:first_done] + [0. for i in range(2 - first_done)]
        if first_done < len(rewards) - 1:
            use_next = False
        if indices + 2 <= end:
            next_state = self.state_memory[indices + 2] 
        else:
            next_state = self.state_memory[start + (indices + 2 - end)]
        return torch.Tensor(n_rewards), torch.Tensor(next_state), torch.Tensor([use_next])
        
        
class PrioritizedReplayMemory:
    def __init__(self, input_dims, max_mem, batch_size, n_envs, env = "FG", size = 20):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mem_size = max_mem
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.n_envs = n_envs
        self.dem_pivot = 0
        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)                                  
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.gammas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.landas_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.dones_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.temporal_errors = np.full(self.mem_size, 1., dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_transition_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), done, gamma, landa)
    
    def store_transition_indiv(self, state, action, reward, state_,done, gamma, landa):
        index = (self.mem_cntr ) % (self.mem_size - self.dem_pivot) 
        self.state_memory[index + self.dem_pivot] = state
        self.action_memory[index + self.dem_pivot] = action
        self.reward_memory[index + self.dem_pivot] = reward
        self.new_state_memory[index + self.dem_pivot] = state_
        self.gammas_memory[index + self.dem_pivot] = gamma
        self.landas_memory[index + self.dem_pivot] = landa
        self.dones_memory[index + self.dem_pivot] = done
        self.temporal_errors[index + self.dem_pivot] = max(self.temporal_errors, default=1.)
        self.mem_cntr += 1

    def store_dem(self, state, action, reward, state_, done, gamma, landa):
        for i in range(state.shape[0]):
            self.store_dem_indiv(state[i].cpu(), action[i].cpu(), reward[i].cpu(), state_[i].cpu(), done, gamma, landa)
    
    def store_dem_indiv(self, state, action, reward, state_,done, gamma, landa):
        index = self.dem_pivot % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.gammas_memory[index] = gamma
        self.landas_memory[index] = landa
        self.dones_memory[index] = done
        self.temporal_errors[index] = max(self.temporal_errors, default=1.)
        self.dem_pivot += 1

    def get_probs(self, priority_scale=1.0):
        max_mem = min(self.mem_cntr + self.dem_pivot, self.mem_size)
        scaled_probs = np.array(self.temporal_errors[:max_mem] ** priority_scale) 
        probs = scaled_probs / sum(scaled_probs)
        return probs
    
    def get_importance(self, probs):
        max_mem = min(self.mem_cntr + self.dem_pivot, self.mem_size)
        importance = (1/max_mem)*(1/probs)
        norm_importance = importance/max(importance)
        return norm_importance

    def set_priority(self, indices, errors, offset = 0.1):
        for i, j in zip(indices, errors.tolist()):
            self.temporal_errors[i] = abs(j) + offset

    def sample_memory(self):
        max_mem = min(self.mem_cntr + self.dem_pivot, self.mem_size)
        probs = self.get_probs()
        batch = np.random.choice(max_mem, self.batch_size,
                                 replace=False, p=probs)
        states = self.state_memory[batch]
        new_states = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        gammas = self.gammas_memory[batch]
        landas = self.landas_memory[batch]
        dones = self.dones_memory[batch]
        importance = self.get_importance(probs[batch])
        dem = batch < self.dem_pivot
        return batch, torch.Tensor(states).to(self.device),  torch.Tensor(actions).to(self.device),  torch.Tensor(rewards).to(self.device),  torch.Tensor(new_states).to(self.device), torch.Tensor(gammas).to(self.device), torch.Tensor(landas).to(self.device), torch.Tensor(dones).bool().to(self.device), torch.Tensor(importance).to(self.device), torch.Tensor(dem).to(self.device)

    def get_all(self):
        max_mem = min(self.mem_cntr, self.mem_size)
        return torch.Tensor(self.state_memory[:max_mem]).to(self.device),  torch.Tensor(self.action_memory[:max_mem]).to(self.device),  torch.Tensor(self.reward_memory[:max_mem]).to(self.device),  torch.Tensor(self.new_state_memory[:max_mem]).to(self.device), torch.Tensor(self.gammas_memory[:max_mem]).to(self.device), torch.Tensor(self.landas_memory[:max_mem]).to(self.device), torch.Tensor(self.dones_memory[:max_mem]).bool().to(self.device)

    def get_n_steps(self, indices):
        dem = indices < self.dem_pivot
        rewards = []
        next_states = []
        use_next = []
        for i in range(len(indices)):
            r, next_s, use_next_ = self.get_n_steps_indiv(indices[i], dem[i])
            rewards.append(r)
            next_states.append(next_s)
            use_next.append(use_next_)
        return torch.stack(rewards).to(self.device), torch.stack(next_states).to(self.device), torch.stack(use_next).to(self.device)
    
    def get_n_steps_indiv(self, indices, dem):
        if dem:
            start = 0
            end = self.dem_pivot - 1 
        else:
            start = self.dem_pivot
            end = min(self.mem_cntr + self.dem_pivot, self.mem_size) - 1
        rewards = list(self.reward_memory[indices:min(indices + 2, end)])
        dones = list(self.dones_memory[indices:min(indices + 2, end)])
        if len(rewards) < 2:
            rewards = rewards + list(self.reward_memory[start:min(start + (2 - (len(rewards))), end)])
            dones = dones + list(self.dones_memory[start:min(start + (2 - (len(rewards))), end)])
        first_done = 0
        for i in dones:
            if i:
                break
            first_done += 1
        use_next = True
        n_rewards = rewards[:first_done] + [0. for i in range(2 - first_done)]
        if first_done < len(rewards) - 1:
            use_next = False
        if indices + 2 <= end:
            next_state = self.state_memory[indices + 2] 
        else:
            next_state = self.state_memory[start + (indices + 2 - end)]
        return torch.Tensor(n_rewards), torch.Tensor(next_state), torch.Tensor([use_next])