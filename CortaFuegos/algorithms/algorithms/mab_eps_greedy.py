import numpy as np
import itertools
from enviroment.utils.final_reward import write_firewall_file, generate_reward
import random
class MAB_eps_greedy:
    def __init__(self, size, c = 2, n_sims = 50, burn_value = 10, eps = 0.1):
        self.size = size
        self.c = c
        self.n_sims = n_sims
        self.burn_value = burn_value
        self.eps = eps
        combinations = list(itertools.product([0, -1], repeat=self.size**2))
        self.action_space = np.asarray(combinations).reshape((len(combinations),self.size, self.size))
        self.Q = np.zeros(len(combinations))
        self.N = np.zeros(len(combinations)) + 1e-5

    def simulate_action(self):
        if self.eps < random.uniform(0, 1):
            action = random.randint(0, len(self.action_space) - 1)
        else:
            action = np.argmax(self.Q)
        solution = self.action_space[action]
        write_firewall_file(solution)
        reward = generate_reward(self.n_sims, self.size)*self.burn_value + solution.sum()
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (reward - self.Q[action])/self.N[action]
        return reward, solution
