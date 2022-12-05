from cmath import inf
from distutils.command.sdist import sdist
from termios import N_MOUSE
import torch
import random
import itertools
import numpy as np
class Q_Table:
    def __init__(self, size, alpha = 1e-4, gamma = 0.99, n_envs = 8, epsilon = 0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_envs = n_envs
        self.n_states = 0
        self.q_table = {}
        self.n_steps = int((self.size**2)*0.05)
        self.action_state = {}
        self.n_states_step = []
        self.create_table()

    def create_table(self):
        forbidden = int((self.size**2)*0.05)//2
        for i in range(self.n_steps + 1):
            combinations = list(itertools.combinations((j for j in range(forbidden, self.size**2)), i))
            self.n_states_step.append(len(combinations))
            for j in range(len(combinations)):
                self.action_state[(i,j)] = []
                state = torch.zeros((self.size, self.size)).to(self.device)
                for c in combinations[j]:
                    l = c // self.size 
                    m = c % self.size
                    state[l,m] = 1
                next_state_comb = list(set(i for i in range(forbidden, self.size**2)) - set(combinations[j]))
                for action in next_state_comb:
                    self.q_table[(i, j, action)] = [0, state]
                    self.action_state[(i,j)].append(action)
                    self.n_states += 1
        print(self.n_states)
    def find_state_indiv(self, state, step):
        n = 0
        for i in range(int(self.n_states_step[step])):
            n+=1
            if torch.equal(self.q_table[(step, i, self.action_state[(step,i)][0])][1], state):
                break
        return n-1
    
    def find_state(self, state, step):
        n_states = []
        for i in range(self.n_envs):
            n_states.append(self.find_state_indiv(state[i], step))
        return n_states
    
    def pick_greedy_action_indiv(self, n_state, step):
        if self.epsilon < random.uniform(0, 1):
                action = random.choice(self.action_state[(step, n_state)])
                return action
        else:
            max_a_value = -inf
            max_action = None
            for a in self.action_state[(step, n_state)]:
                if self.q_table[(step, n_state, a)][0] >= max_a_value:
                    max_a_value = self.q_table[(step, n_state, a)][0]
                    max_action = a
            return max_action
        
    def pick_greedy_action(self, n_states, step):
        actions = []
        for n in n_states:
            actions.append(self.pick_greedy_action_indiv(n, step))
        return torch.Tensor(actions).to(self.device)

    def update_table_indiv(self, n_state, step, action, next_state, reward):
        n_next_state = self.find_state_indiv(next_state, step)
        max_a_value = -inf
        max_action = None
        for a in self.action_state[(step + 1, n_next_state)]:
            if self.q_table[(step + 1, n_next_state, a)][0] >= max_a_value:
                max_a_value = self.q_table[(step + 1, n_next_state, a)][0]
                max_action = a
        self.q_table[(step, n_state, action)][0] = self.q_table[(step, n_state, action)][0] + self.alpha*(reward + self.gamma*self.q_table[(step + 1, n_next_state, max_action)][0]-self.q_table[(step, n_state, action)][0])

    def update_table(self, n_states, step, actions, next_states, rewards):
        for i in range(self.n_envs):
            self.update_table_indiv(n_states[i], step, actions[i].item(), next_states[i], rewards[i].item())

    def max_action(self, state, step):
        n_state = self.find_state_indiv(state, step)
        max_a_value = -inf
        max_action = None
        for a in self.action_state[(step, n_state)]:
            if self.q_table[(step, n_state,a)][0] >= max_a_value:
                max_a_value = self.q_table[(step, n_state,a)][0]
                max_action = a
        return torch.Tensor([max_action]).to(self.device)

class Q_Table_2:
    def __init__(self, size, alpha = 1e-4, gamma = 0.99, n_envs = 8, epsilon = 0.1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.size = size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_envs = n_envs
        self.n_states = 0
        self.n_steps = (self.size//2)*2 + 1
        self.n_states_step = np.zeros(self.n_steps)
        self.q_table = {}
        self.create_table()
        
    def create_table(self):
        ### Initial state
        state = torch.zeros((2, self.size, self.size)).to(self.device)
        state[0][0,0] = 1
        state[0][1,0] = 1
        state[0][0,1] = 1
        state[0][1,1] = 1
        state[1] = 2/101
        for i in range(16):
            self.q_table[(0, 0, i)] = [random.uniform(0, 1), state]
            self.n_states += 1
        self.n_states_step[0] += 1
        ### First step
        combinations = list(itertools.product([1, 2/101], repeat=4))
        state_combinations = np.asarray(combinations).reshape((len(combinations), 4))
        for i in range(len(combinations)):
            state = torch.zeros((2, self.size, self.size)).to(self.device)
            if self.size > 2:
                state[0][2,0] = 1
                state[0][3,0] = 1
                state[0][2,1] = 1
                state[0][3,1] = 1
                state[1][0,0] = state_combinations[i][0]
                state[1][1,0] = state_combinations[i][1]
                state[1][0,1] = state_combinations[i][2]
                state[1][1,1] = state_combinations[i][3]
            else:
                state[0][0,0] = 0
                state[0][1,0] = 0
                state[0][0,1] = 0
                state[0][1,1] = 0
                state[1][0,0] = state_combinations[i][0]
                state[1][1,0] = state_combinations[i][1]
                state[1][0,1] = state_combinations[i][2]
                state[1][1,1] = state_combinations[i][3]
            for j in range(16):
                self.n_states += 1
                if self.size > 2:
                    self.q_table[(1, i, j)] = [random.uniform(0, 1), state]
                else:
                    self.q_table[(1, i, j)] = [0, state]
            self.n_states_step[1] += 1
        if self.size == 2:
            return
        ### Second step
        combinations = list(itertools.product([1, 2/101], repeat=8))
        state_combinations = np.asarray(combinations).reshape((len(combinations), 8))
        for i in range(len(combinations)):
            state = torch.zeros((2, self.size, self.size)).to(self.device)
            state[0][0,2] = 1
            state[0][1,2] = 1
            state[0][0,3] = 1
            state[0][1,3] = 1
            state[1][0,0] = state_combinations[i][0]
            state[1][1,0] = state_combinations[i][1]
            state[1][0,1] = state_combinations[i][2]
            state[1][1,1] = state_combinations[i][3]
            state[1][2,0] = state_combinations[i][4]
            state[1][2,1] = state_combinations[i][5]
            state[1][3,0] = state_combinations[i][6]
            state[1][3,1] = state_combinations[i][7]
            for j in range(16):
                self.q_table[(2, i, j)] = [random.uniform(0, 1), state]
                self.n_states += 1
            self.n_states_step[2] += 1
        ### Third step
        combinations = list(itertools.product([1, 2/101], repeat=12))
        state_combinations = np.asarray(combinations).reshape((len(combinations), 12))
        for i in range(len(combinations)):
            state = torch.zeros((2, self.size, self.size)).to(self.device)
            state[0][2,2] = 1
            state[0][3,2] = 1
            state[0][2,3] = 1
            state[0][3,3] = 1
            state[1][0,0] = state_combinations[i][0]
            state[1][1,0] = state_combinations[i][1]
            state[1][0,1] = state_combinations[i][2]
            state[1][1,1] = state_combinations[i][3]
            state[1][2,0] = state_combinations[i][4]
            state[1][2,1] = state_combinations[i][5]
            state[1][3,0] = state_combinations[i][6]
            state[1][3,1] = state_combinations[i][7]
            state[1][0,2] = state_combinations[i][8]
            state[1][1,2] = state_combinations[i][9]
            state[1][0,3] = state_combinations[i][10]
            state[1][1,3] = state_combinations[i][11]
            for j in range(16):
                self.q_table[(3, i, j)] = [random.uniform(0, 1), state]
                self.n_states += 1
            self.n_states_step[3] += 1
        ### Fourth step

        combinations = list(itertools.product([1, 2/101], repeat=16))
        state_combinations = np.asarray(combinations).reshape((len(combinations), 16))
        for i in range(len(combinations)):
            state = torch.zeros((2, self.size, self.size)).to(self.device)
            state[1][0,0] = state_combinations[i][0]
            state[1][1,0] = state_combinations[i][1]
            state[1][0,1] = state_combinations[i][2]
            state[1][1,1] = state_combinations[i][3]
            state[1][2,0] = state_combinations[i][4]
            state[1][2,1] = state_combinations[i][5]
            state[1][3,0] = state_combinations[i][6]
            state[1][3,1] = state_combinations[i][7]
            state[1][0,2] = state_combinations[i][8]
            state[1][1,2] = state_combinations[i][9]
            state[1][0,3] = state_combinations[i][10]
            state[1][1,3] = state_combinations[i][11]
            state[1][0,2] = state_combinations[i][12]
            state[1][1,2] = state_combinations[i][13]
            state[1][0,3] = state_combinations[i][14]
            state[1][1,3] = state_combinations[i][15]
            for j in range(16):
                self.q_table[(4, i, j)] = [0, state]
            self.n_states_step[4] += 1
       
    def find_state_indiv(self, state, step):
        n = 0
        for i in range(int(self.n_states_step[step])):
            n+=1
            if torch.equal(self.q_table[(step, i, 0)][1], state):
                break
        return n-1
    
    def find_state(self, state, step):
        n_states = []
        for i in range(self.n_envs):
            n_states.append(self.find_state_indiv(state[i], step))
        return n_states

    def pick_greedy_action_indiv(self, n_state, step):
        if self.epsilon < random.uniform(0, 1):
                action = (random.randint(0, 15))
                return action
        else:
            max_a_value = -inf
            max_action = None
            for a in range(16):
                if self.q_table[(step, n_state, a)][0] >= max_a_value:
                    max_a_value = self.q_table[(step, n_state, a)][0]
                    max_action = a
            return max_action
        
    def pick_greedy_action(self, n_states, step):
        actions = []
        for n in n_states:
            actions.append(self.pick_greedy_action_indiv(n, step))
        return torch.Tensor(actions).to(self.device)

    def update_table_indiv(self, n_state, step, action, next_state, reward):
        n_next_state = self.find_state_indiv(next_state, step)
        max_a_value = -inf
        max_action = None
        for a in range(16):
            if self.q_table[(step + 1, n_next_state, a)][0] >= max_a_value:
                max_a_value = self.q_table[(step + 1, n_next_state, a)][0]
                max_action = a
        self.q_table[(step, n_state, action)][0] = self.q_table[(step, n_state, a)][0] + self.alpha*(reward + self.gamma*self.q_table[(step + 1, n_next_state, max_action)][0]-self.q_table[(step, n_state, action)][0])

    def update_table(self, n_states, step, actions, next_states, rewards):
        for i in range(self.n_envs):
            self.update_table_indiv(n_states[i], step, actions[i].item(), next_states[i], rewards[i].item())

    def max_action(self, state, step):
        n_state = self.find_state_indiv(state, step)
        max_a_value = -inf
        max_action = None
        for a in range(16):
            if self.q_table[(step, n_state,a)][0] >= max_a_value:
                max_a_value = self.q_table[(step, n_state,a)][0]
                max_action = a
        return torch.Tensor([max_action]).to(self.device)
