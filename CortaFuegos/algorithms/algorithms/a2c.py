import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
from algorithms.utils.plot_progress import plot_prog
from torch.utils.data import Dataset, DataLoader
from algorithms.utils.plot_progress import plot_moving_av, plot_loss

# reinforce with baseline algorithm:
def a2c(env, net, episodes, env_version, net_version, plot_episode, n_envs = 8, alpha = 1e-4, gamma = 0.99, beta = 0.01, instance = "sub20x20", test = False, window = 10):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            state_c = state.clone()
            mask = env.generate_mask()
            policy, value, entropy = net.forward(state_c, mask = mask)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            next_mask = env.generate_mask()
            _, value_next_state, _ = net.forward(next_state_c, mask = next_mask)
            I *= gamma
            net.zero_grad()
            target = reward + gamma * value_next_state
            critic_loss = F.mse_loss(value, target)
            advantage = (target - value)
            log_probs = torch.log(policy + 1e-6)
            action_log_probs = log_probs.gather(1, action.squeeze(0))
            actor_loss = torch.sum(- I * action_log_probs * advantage - beta*entropy)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
            ep_return += reward
            state = next_state
            I *= gamma
        stats["Loss"].append(total_loss.item())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version ,"a2c", env.size, instance, test)
    plot_moving_av(env.envs[0], stats["Returns"], episodes, env_version, net_version, "a2c", window = window, instance = instance, test = test)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "a2c", test)
    return stats
