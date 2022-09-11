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
# A2C algorithm:
def batched_reinforce_baseline(env, net, episodes, version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.01, update_steps = 10):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    step_data = []
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            policy, value = net.forward(copy.copy(state))
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            _, value_next_state = net.forward(copy.copy(next_state))
            I *= gamma
            step_data.append([action, reward, policy, value, value_next_state, I])
            state = next_state
            ep_return += reward
        if episode % 10 == 0:
            data = DataLoader(step_data, len(step_data), shuffle=False)
            for action_t, reward_t, policy_t, value_t, value_next_state_t, discounts in data:
                now = datetime.datetime.now()
                target = reward_t + gamma * value_next_state_t
                net.zero_grad()
                critic_loss = F.mse_loss(value_t.squeeze(), target)
                # critic_loss.backward(retain_graph=True)
                # now = datetime.datetime.now()
                advantage = (target - value_t).squeeze()
                log_probs = torch.log(policy_t.squeeze() + 1e-6)
                action_log_probs = log_probs.gather(1, action_t.squeeze().unsqueeze(1))
                entropy = -torch.sum(policy_t.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
                actor_loss = torch.sum(- discounts * action_log_probs * advantage - 0.02*entropy)
                # actor_loss.backward()
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                # later = datetime.datetime.now()
                optimizer.step()
                # print(later-now)
            stats["Actor Loss"].append(actor_loss.item())
            stats["Critic Loss"].append(critic_loss.item())
            step_data = []
        
        stats["Returns"].append(ep_return.mean().item())
        if episode in plot_episode:
            plot_prog(env, episode, net, version ,"figures", "batched_reinforce_baseline" )
    return stats

