import numpy as np
import torch
import torchvision
import torchvision.transforms.functional as F
import itertools
from enviroment.utils.final_reward import write_firewall_file, generate_reward
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
import json
class MAB_eps_greedy:
    def __init__(self, size, c = 2, n_sims = 50, burn_value = 10, eps = 0.1):
        self.size = size
        self.c = c
        self.n_sims = n_sims
        self.burn_value = burn_value
        self.eps = eps
    
    def simulate_action(self):
        if self.eps < random.uniform(0, 1):
            action = random.randint(0, len(self.action_space) - 1)
        else:
            action = np.argmax(self.Q)
        solution = self.action_space[action]
        if self.size < 20:
            up_solution = self.up_scale(solution, 20)
            write_firewall_file(up_solution)
        else:
            write_firewall_file(solution)
        reward = generate_reward(self.n_sims, self.size)*self.burn_value + solution.sum()
        self.N[action] += 1
        self.Q[action] = self.Q[action] + (reward - self.Q[action])/self.N[action]
        return reward, solution

    def up_scale(self, arr, size):
        t = torch.from_numpy(arr).unsqueeze(0)
        t_resized = F.resize(t, size, interpolation = torchvision.transforms.InterpolationMode.NEAREST).squeeze(0)
        return t_resized
        
    def opt_solution(self):
        action = np.argmax(self.Q)
        solution = self.action_space[action]
        return solution

    

class MAB_eps_greedy_MG(MAB_eps_greedy):
    def __init__(self, size, c = 2, n_sims = 50, burn_value = 10, eps = 0.1):
        super().__init__(size, c, n_sims, burn_value, eps)
        combinations = list(itertools.product([0, -1], repeat=self.size**2))
        self.action_space = np.asarray(combinations).reshape((len(combinations),self.size, self.size))
        self.Q = np.zeros(len(combinations))
        self.N = np.zeros(len(combinations)) + 1e-5

    

    
class MAB_eps_greedy_FG(MAB_eps_greedy):
    def __init__(self, size, c = 2, n_sims = 50, burn_value = 10, eps = 0.1):
        super().__init__(size, c, n_sims, burn_value, eps)
        self.n_marks = int((self.size**2)*0.05)
        self.n_states = 0
        self.action_space = {}
        forbidden = int((self.size**2)*0.05)//2
        combinations = list(itertools.combinations((j for j in range(forbidden,self.size**2)), self.n_marks))
        for i in range(len(combinations)):
                self.n_states += 1
                state = np.zeros((self.size, self.size))
                for c in combinations[i]:
                    l = c // self.size 
                    m = c % self.size
                    state[l,m] = -1
                self.action_space[i] = state
        self.Q = np.zeros(len(combinations))
        self.N = np.zeros(len(combinations)) + 1e-5


def mab_greedy(env, size, episodes, epsilon, window, instance):
    if env.envs[0].get_name() == "moving_grid":
        mab = MAB_eps_greedy_MG(size, eps = epsilon)
        path = f"figures_tuning/moving_grid/mab/{instance}/epsilon_greedy/sub{size}x{size}"
    else:
        mab = MAB_eps_greedy_FG(size, eps = epsilon)
        path = f"figures_tuning/full_grid/mab/{instance}/epsilon_greedy/sub{size}x{size}"
    rewards = []
    for i in tqdm(range(episodes)):
        reward, solution = mab.simulate_action()
        rewards.append(reward)
    with open(f"data/{env.envs[0].name}/v1/{instance}/sub{size}x{size}/mab_greedy/stats_{episodes}.json", "w+") as write_file:
        json.dump(rewards, write_file, indent=4)
    ret = np.cumsum(rewards, dtype=float)
    ret[window:] = ret[window:] - ret[:-window]
    figure2 = plt.figure()
    plt.clf()
    plt.plot(np.arange(len(ret[window - 1:])) + window, ret[window - 1:] / window)
    plt.xlabel("Episode") 
    plt.ylabel("Average return") 
    plt.title(f"Average returns for {episodes} episodes")
    plt.savefig(f"{path}/returns_{episodes}")
    plt.show() 
    figure5 = plt.figure()
    plt.imshow(mab.opt_solution())
    plt.colorbar()
    plt.title(f"Agent's trajectory after {episodes} episodes")
    plt.savefig(f"{path}/trajectory_{episodes}")
    plt.show()

    