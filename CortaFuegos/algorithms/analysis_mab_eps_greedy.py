from algorithms.mab_eps_greedy import MAB_eps_greedy
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

_, size_, episodes_, eps_, window_, instance= sys.argv
size = int(size_)
episodes = int(episodes_)
window = int(window_)
eps = float(eps_)
mab = MAB_eps_greedy(size, eps_)
rewards = []
for i in tqdm(range(episodes)):
    reward, solution = mab.simulate_action()
    rewards.append(reward)

ret = np.cumsum(rewards, dtype=float)
ret[window:] = ret[window:] - ret[:-window]
figure2 = plt.figure()
plt.clf()
plt.plot(np.arange(len(ret[window - 1:])) + window, ret[window - 1:] / window)
plt.xlabel("Episode") 
plt.ylabel("Average return") 
plt.title(f"Average returns for {episodes} episodes")
plt.savefig(f"figures_tuning/mab/epsilon_greedy/{instance}/returns_{episodes}.png")
plt.show() 

figure5 = plt.figure()
plt.imshow(solution)
plt.colorbar()
plt.title(f"Agent's trajectory after {episodes} episodes")
plt.savefig(f"figures_tuning/mab/epsilon_greedy/{instance}/trayectory_{episodes}.png")
plt.show()