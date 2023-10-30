import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
import os
import torch
def plot_prog(env,episodes, net, env_version, net_version, algorithm, size, instance, test = False, params = {}):
    state = env.reset()
    if test:
        path = f"figures_tuning/{env.get_name()}"
    else:
        path = f"figures/{env.get_name()}"
    try:
        os.makedirs(f"{path}/{env_version}/{instance}/sub{size}x{size}/{net_version}/{algorithm}")
    except OSError as error:  
        print(error)   
    probs_dir = f"{path}/{env_version}/{instance}/{net_version}/{algorithm}/{episodes}_ep/probabilities/post_train/"
    probs_dir_list = os.listdir(probs_dir)
    for i in probs_dir_list:
        os.remove(probs_dir+i)
    for i in range(env.get_episode_len()):
        if algorithm == "ddqn":
            q_pred, value = net.forward(state)
            selected = net.sample(q_pred, state.unsqueeze(0))
        else:
            q_pred = net.forward(state)
            selected = net.sample(q_pred, state.unsqueeze(0))
        state, _, done = env.step(selected)
        if done:
            mat = env._space[0].reshape(size,size).to('cpu').numpy()
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory")
            plt.savefig(f"{path}/{env_version}/{instance}/sub{size}x{size}/{net_version}/{algorithm}/{episodes}_ep/trajectory/trajectory.png")
            plt.show()

def plot_trayectory_probs(env,episodes, net, env_version, net_version, algorithm, size, instance, test = False, params = {}):
    params_dir = ""
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    state = env.reset()
    if test:
        path = f"figures_tuning/{env.get_name()}"
    else:
        path = f"figures/{env.get_name()}"
    try:
        os.makedirs(f"{path}/{env_version}/{instance}/sub{size}x{size}/{net_version}/{algorithm}")
    except OSError as error:  
        print(error)
    for i in range(env.get_episode_len()):
        if algorithm == "ddqn":
            q_pred, value = net.forward(state)
            selected = net.sample(q_pred, state.unsqueeze(0))
        else:
            q_pred = net.forward(state)
            selected = net.sample(q_pred, state.unsqueeze(0))
        state, _, done = env.step(selected)
        if done:
            mat = env._space[0].reshape(size,size).to('cpu').numpy()
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory")
            plt.savefig(f"{path}/{env_version}/{instance}/sub{size}x{size}/{net_version}/{algorithm}/trajectory_episodes={episodes}_{params_dir}.png")
            plt.show()

def plot_moving_av(env, returns, episodes, env_version, net_version, algorithm, window = 100, drl = True, test = False, params = {}, instance = "sub20x20"):
    params_dir = ""
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    if not test:
        base_dir = f"figures/{env.get_name()}"
    else:
        base_dir = f"figures_tuning/{env.get_name()}"
    try:
        os.makedirs(f"{base_dir}/{env_version}/{instance}/sub{env.size}x{env.size}/{net_version}/{algorithm}")
    except OSError as error:  
        print(error)
    if drl:
        if len(returns) < window:
            figure2 = plt.figure()
            plt.clf()
            plt.plot(np.arange(len(returns)), returns)
            plt.xlabel("Episode") 
            plt.ylabel("Returns") 
            plt.title(f"Returns for {episodes} episodes")
            plt.savefig(f"{base_dir}/{env_version}/{instance}/sub{env.size}x{env.size}/{net_version}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
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
            plt.savefig(f"{base_dir}/{env_version}/{instance}/sub{env.size}x{env.size}/{net_version}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
            plt.show() 
    else:
        if len(returns) < window:
            figure2 = plt.figure()
            plt.clf()
            plt.plot(np.arange(len(returns)), returns)
            plt.xlabel("Episode") 
            plt.ylabel("Returns") 
            plt.title(f"Returns for {episodes} episodes")   
            plt.savefig(f"{base_dir}/{env_version}/{instance}/sub{env.size}x{env.size}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
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
            plt.savefig(f"{base_dir}/{env_version}/{instance}/sub{env.size}x{env.size}/{algorithm}/returns_episodes={episodes}_{params_dir}.png")
            plt.show() 

def plot_loss(env, loss, episodes, env_version, instance, net_version, algorithm, test, params = {}):
    params_dir = ""
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    if test:
        path = f"figures_tuning/{env.get_name()}"
    else:
        path = f"figures/{env.get_name()}"
    try:
        os.makedirs(f"{path}/{env_version}/{instance}/sub{env.size}x{env.size}/{net_version}/{algorithm}")
    except OSError as error:  
        print(error)
    figure3 = plt.figure()
    plt.clf()
    plt.plot([i for i in range(len(loss))], loss)
    plt.xlabel("Episode") 
    plt.ylabel("Loss") 
    plt.title(f"Loss for {episodes} episodes")
    plt.savefig(f"{path}/{env_version}/{instance}/sub{env.size}x{env.size}/{net_version}/{algorithm}/loss_episodes={episodes}_{params_dir}.png")
    plt.show() 



