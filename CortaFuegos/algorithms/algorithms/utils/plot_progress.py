import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
import os

def plot_prog(env,episodes, net, env_version, net_version, algorithm, size, instance, test = False):
    state = env.reset()
    if test:
        path = f"figures_tuning/{env.get_name()}"
    else:
        path = f"figures/{env.get_name()}"
    probs_dir = f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/"
    probs_dir_list = os.listdir(probs_dir)
    for i in probs_dir_list:
        os.remove(probs_dir+i)
    for i in range(env.get_episode_len()):
        mat = env._space[0].reshape(size,size).to('cpu').numpy()
        mask = env.generate_mask().bool()
        if algorithm == "reinforce":
            a, _ = net.forward(state, mask)
        else:
            a,_, _ = net.forward(state, mask)
        f2 = plt.figure()
        plt.clf()
        plt.bar(np.arange(env.get_action_space().shape[0]), a.detach().to('cpu').numpy().squeeze())
        plt.xlabel("Actions") 
        plt.ylabel("Action Probability") 
        plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
        plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
        plt.show()
        selected = a.multinomial(1).detach()
        state, _, done = env.step(selected)
        if done:
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory")
            plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/trajectory/trajectory.png")
            plt.show()

def plot_moving_av(env, returns, episodes, env_version, net_version, algorithm, window = 100, drl = True, test = False, params = {}, instance = "sub20x20"):
    params_dir = ""
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    if not test:
        base_dir = f"figures/{env.get_name()}"
    else:
        base_dir = f"figures_tuning/{env.get_name()}"
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

def plot_loss(env, loss, episodes, env_version, instance, net_version, algorithm, test):
    if test:
        path = f"figures_tuning/{env.get_name()}"
    else:
        path = f"figures/{env.get_name()}"
    figure3 = plt.figure()
    plt.clf()
    plt.plot([i for i in range(episodes)], loss)
    plt.xlabel("Episode") 
    plt.ylabel("Loss") 
    plt.title(f"Loss for {episodes} episodes")
    plt.savefig(f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/loss.png")
    plt.show() 



