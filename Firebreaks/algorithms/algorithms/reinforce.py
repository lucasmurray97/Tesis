import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from algorithms.utils.plot_progress import plot_prog
from torch.utils.data import Dataset, DataLoader
from algorithms.utils.plot_progress import plot_moving_av, plot_loss, plot_trayectory_probs
from algorithms.utils.replay_buffer import ReplayMemory
import json
import copy
# Reinforce Algorithm:
def reinforce(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.1, n_envs = 8, instance = "sub20x20", test = False, window = 10, epochs = 10, batch_size = 64, demonstrate = True, n_dem = 10, combined = False, max_mem = 1000, target_update = 1):
    optim = AdamW(net.parameters(), lr = alpha)
    env_shape = env.env_shape
    ep_len = env.envs[0].get_episode_len()
    memory = ReplayMemory(env_shape, max_mem=max_mem, batch_size=batch_size, demonstrate=demonstrate, n_dem=n_dem, combined=combined, temporal=True, env="FG", version=env_version, size=env_shape[1],n_envs=n_envs, gamma = gamma, landa = 1.)
    stats = {"Loss": [], "Returns": []}
    target_net = copy.deepcopy(net)
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        step = 0
        while not done:
            state_c = state.clone()
            policy, _, _ = target_net.forward(state_c)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            I *= gamma
            memory.buffer.store_transition(state, action, reward, next_state, step, done, I, 1.)
            state = next_state
            ep_return += reward
            step = step + 1
        if memory.is_sufficient():
            loss_acum = 0
            state_t, action_t, reward_t, _, discounts_t, _ = memory.buffer.sample_memory()
            for e in range(epochs):
                net.zero_grad()
                policy_t, value_t, entropy_t = net.forward(state_t)
                log_probs = torch.log(policy_t + 1e-6)
                action_log_probs = log_probs.gather(1, action_t.unsqueeze(1).type(torch.int64))
                entropy = entropy_t
                discounted_rewards = discounts_t.flip(0) * reward_t
                G = torch.cumsum(discounted_rewards, dim=0)
                total_loss = torch.sum(- (G * action_log_probs * discounts_t) - beta*entropy)
                loss_acum += total_loss
                total_loss.backward()
                optim.step()
            stats["Loss"].append(loss_acum.detach().mean().item())
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version, "reinforce", env.size, instance = instance, test = test)
        stats["Returns"].append(ep_return.mean().item())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
    params = {"alpha": alpha, "gamma": gamma, "beta": beta}
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "reinforce", window = window, instance = instance, test = test, params=params)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "reinforce", test)
    plot_trayectory_probs(env.envs[0], episode, net, env_version, net_version ,"reinforce", env.size, instance, test, params = params)
    params_dir = f"episodes={episodes*n_envs}_"
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    with open(f"data/{env.envs[0].name}/{instance}/{env_version}/{net_version}/reinforce/stats_{params_dir}.json", "w+") as write_file:
        json.dump(stats, write_file, indent=4)
    return stats


