import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
import os

def plot_prog(env,episodes, net, env_version, net_version, algorithm, size, instance, test = False):
    state = env.reset()
    if test:
        path = "figures_tuning"
    else:
        path = "figures"
    probs_dir = f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/"
    probs_dir_list = os.listdir(probs_dir)
    for i in probs_dir_list:
        os.remove(probs_dir+i)
    for i in range((size//2)**2):
        if env_version == "v6" or env_version == "v7" or env_version == "v8":
            mat = env._space[0].reshape(size,size).numpy()
        else: 
            mat = state[0].reshape(size,size).numpy()
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
        plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
        plt.show()
        selected = a.multinomial(1).detach()
        state, done, _ = env.step(selected)
        if i == (size//2)**2-1:
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory")
            plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/trajectory/trajectory.png")
            plt.show()

def plot_moving_av(returns, episodes, env_version, net_version, algorithm, window = 100, drl = True, test = False, params = {}, instance = "sub20x20"):
    params_dir = ""
    for key in params.keys():
            params_dir += key + "=" + str(params[key])
    if not test:
        base_dir = "figures"
    else:
        base_dir = "figures_tuning"
        params_dir = ""
    if drl:
        if len(returns) < window:
            figure2 = plt.figure()
            plt.clf()
            plt.plot(np.arange(len(returns)), returns)
            plt.xlabel("Episode") 
            plt.ylabel("Returns") 
            plt.title(f"Returns for {episodes} episodes")
            plt.savefig(f"{base_dir}/{env_version}/{instance}/{net_version}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
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
            plt.savefig(f"{base_dir}/{env_version}/{instance}/{net_version}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
            plt.show() 
    else:
        if len(returns) < window:
            figure2 = plt.figure()
            plt.clf()
            plt.plot(np.arange(len(returns)), returns)
            plt.xlabel("Episode") 
            plt.ylabel("Returns") 
            plt.title(f"Returns for {episodes} episodes")
            plt.savefig(f"{base_dir}/{env_version}/{instance}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
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
            plt.savefig(f"{base_dir}/{env_version}/{instance}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
            plt.show() 




