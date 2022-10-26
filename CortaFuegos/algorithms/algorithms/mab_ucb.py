import numpy as np
import itertools
from enviroment.utils.final_reward import write_firewall_file, generate_reward
class MAB_UCB:
    def __init__(self, size, c = 2, n_sims = 50, burn_value = 10):
        self.size = size
        self.c = c
        self.t = 1
        self.n_sims = n_sims
        self.burn_value = burn_value
        combinations = list(itertools.product([0, -1], repeat=self.size**2))
        self.action_space = np.asarray(combinations).reshape((len(combinations),self.size, self.size))
        self.Q = np.zeros(len(combinations))
        self.N = np.zeros(len(combinations)) + 1e-5
        self.ucb = self.Q + self.c * np.sqrt(np.log(self.t)/self.N) 
    def simulate_action(self):
        action = np.argmax(self.ucb)
        solution = self.action_space[action]
        write_firewall_file(solution)
        reward = generate_reward(self.n_sims, self.size)*self.burn_value + solution.sum()
        self.Q[action] = self.Q[action] + (reward - self.Q[action])/self.N[action]
        self.N[action] += 1
        self.t += 1
        self.ucb[action] = self.Q[action]+ self.c * np.sqrt(np.log(self.t)/self.N[action]) 
        return reward, solution
