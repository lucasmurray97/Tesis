import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.parallel_firegrid import Parallel_Firegrid
import torch.nn.functional as F
from utils.plot_progress import plot_prog
from torch.utils.data import Dataset, DataLoader
import datetime
import copy
# reinforce with baseline algorithm:
def a2c(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.01):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            policy, value = net.forward(state.clone().unsqueeze(0))
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            _, value_next_state = net.forward(next_state.clone().unsqueeze(0))
            I *= gamma
            target = reward + gamma * value_next_state
            net.zero_grad()
            critic_loss = F.mse_loss(value.squeeze(), target)
            advantage = (target - value).squeeze()
            log_probs = torch.log(policy.squeeze() + 1e-6)
            action_log_probs = log_probs.gather(0, action.squeeze().unsqueeze(0))
            entropy = -torch.sum(policy.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
            actor_loss = torch.sum(- I * action_log_probs * advantage - 0.02*entropy)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
            ep_return += reward
            state = next_state
            I *= gamma
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())
        if episode in plot_episode:
            plot_prog(env, episode, net, env_version, net_version ,"figures", "a2c" )
    return stats
