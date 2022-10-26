from algorithms.mab_ucb import MAB_UCB
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

_, size_, episodes_, instance, window_= sys.argv
size = int(size_)
episodes = int(episodes_)
window = int(window_)
mab = MAB_UCB(size)
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
plt.savefig(f"figures_tuning/mab/ucb/{instance}/returns_{episodes}.png")
plt.show() 

figure5 = plt.figure()
plt.imshow(solution)
plt.colorbar()
plt.title(f"Agent's trajectory after {episodes} episodes")
plt.savefig(f"figures_tuning/mab/ucb/{instance}/trayectory_{episodes}.png")
plt.show()