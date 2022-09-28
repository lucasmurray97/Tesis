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
def reinforce_baseline(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.02, update_steps = 10):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        step_data = []
        while not done:
            state_c = state.clone()
            policy, value = net.forward(state_c.unsqueeze(0))
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            _, value_next_state = net.forward(next_state_c.unsqueeze(0))
            step_data.append([action, reward, policy, value, value_next_state*done, I])
            I *= gamma
            state = next_state
            ep_return += reward
        step_data.reverse()
        data = DataLoader(step_data, len(step_data), shuffle=False)
        for action_t, reward_t, policy_t, value_t, value_next_state_t, discounts in data:
            target = reward_t + gamma * value_next_state_t
            net.zero_grad()
            critic_loss = F.mse_loss(value_t.squeeze(), target)
            log_probs = torch.log(policy_t.squeeze() + 1e-6)
            action_log_probs = log_probs.gather(1, action_t.squeeze().unsqueeze(1))
            entropy = -torch.sum(policy_t.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
            discounted_rewards = discounts.flip(0) * reward_t.squeeze()
            G = torch.cumsum(discounted_rewards, dim=0)
            actor_loss = torch.sum(- (G - value_t) * action_log_probs * discounts - beta*entropy)
            total_loss = critic_loss + actor_loss
            total_loss.backward()
            optimizer.step()
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())
        if episode in plot_episode:
            plot_prog(env, episode, net, env_version, net_version ,"figures", "reinforce_baseline" )
    return stats

