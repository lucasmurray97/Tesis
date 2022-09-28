import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np

def plot_prog(env,episodes, net, env_version, net_version, path,algorithm):
    state = env.reset()
    for i in range(100):
        if env_version == "v6" or env_version == "v7" or env_version == "v8":
            mat = env._space[0].reshape(20,20).numpy()
        else: 
            mat = state[0].reshape(20,20).numpy()
        if algorithm == "reinforce":
            a = net.forward(state)
        else:
            a, _ = net.forward(state.unsqueeze(0))
        f2 = plt.figure()
        plt.clf()
        plt.bar(np.arange(16), a.detach().numpy().squeeze())
        plt.xlabel("Actions") 
        plt.ylabel("Action Probability") 
        plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
        plt.savefig(f"{path}/{env_version}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
        plt.show()
        selected = a.multinomial(1).detach()
        state, done, _ = env.step(selected)
        if i == 99:
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory")
            plt.savefig(f"{path}/{env_version}/{net_version}/{algorithm}/{episodes}_ep/trajectory/trajectory.png")
            plt.show()

def plot_moving_av(returns, episodes, env_version, net_version,algorithm, window = 100):
    if len(returns) < window:
        figure2 = plt.figure()
        plt.clf()
        plt.plot(np.arange(len(returns)), returns)
        plt.xlabel("Episode") 
        plt.ylabel("Returns") 
        plt.title(f"Returns for {episodes} episodes")
        plt.savefig(f"figures/{env_version}/{net_version}/{algorithm}/returns.png")
        plt.show() 
    else:
        ret = np.cumsum(returns, dtype=float)
        ret[window:] = ret[window:] - ret[:-window]
        figure2 = plt.figure()
        plt.clf()
        plt.plot(np.arange(len(ret[window - 1:])) + window, ret[window - 1:] / window)
        plt.xlabel("Episode") 
        plt.ylabel("Average return") 
        plt.title(f"Average returns for {episodes} episodes")
        plt.savefig(f"figures/{env_version}/{net_version}/{algorithm}/returns.png")
        plt.show() 




