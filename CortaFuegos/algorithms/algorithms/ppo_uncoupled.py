import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
from utils.plot_progress import plot_prog
from torch.utils.data import DataLoader, RandomSampler

from cnn_policy_value_v2 import CNN
from enviroment.moving_grid.firegrid_v4 import FireGrid_V4
from torch.optim import AdamW
import datetime
import copy
# reinforce with baseline algorithm:
def reinforce_baseline(env, net, episodes, version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.01, update_steps = 1, clip = 0.2, epochs = 10, gae_lambda = 0.95):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    step_data = []
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            policy, value = net.forward(state.clone())
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            _, value_next_state = net.forward(next_state.clone())
            I *= gamma
            step_data.append([state, action, reward, policy, value, value_next_state, I])
            state = next_state
            ep_return += reward
        data = DataLoader(step_data, len(step_data)/10, shuffle=True)
        if episode % 5 == 0:
            for e in range(5):
                # print(e)
                state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts = next(iter(data))
                net.zero_grad()
                target = reward_t + gamma * value_next_state_t
                critic_loss = F.mse_loss(value_t.squeeze(), target)
                advantage = (target - value_t).squeeze()
                new_probs, _ = net.forward(state_t.clone())
                new_log_probs = torch.log(new_probs.squeeze() + 1e-6)
                log_probs = torch.log(policy_t.squeeze() + 1e-6)
                action_log_probs = log_probs.gather(0, action_t.squeeze().unsqueeze(0))
                new_action_log_probs = new_log_probs.gather(0, action_t.squeeze().unsqueeze(0))
                prob_ratio = torch.exp(new_action_log_probs)/torch.exp(action_log_probs)
                weighted_probs = prob_ratio * advantage
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip, 1+clip)*advantage
                entropy = -torch.sum(policy_t.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
                actor_loss = torch.sum(- discounts * torch.minimum(weighted_probs, weighted_clipped_probs) - 0.02*entropy)
                total_loss = actor_loss + critic_loss
                total_loss.backward()
                optimizer.step()
        step_data = []
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())
        if episode in plot_episode:
            plot_prog(env, episode, net, version ,"figures", "reinforce_baseline" )
    return stats

torch.autograd.set_detect_anomaly(True)
env = FireGrid_V4(20, burn_value=10, n_sims=50)
net = CNN()
gamma = 0.99
alpha = 1e-4
clip = 0.2
optimizer = AdamW(net.parameters(), lr = alpha)
stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
step_data = []
for episode in tqdm(range(1, 10 + 1)):
    state = env.reset()
    done = False
    ep_return  = 0
    I = 1.
    while not done:
        policy, value = net.forward(state.clone())
        action = policy.multinomial(1)
        next_state, reward, done = env.step(action.detach())
        _, value_next_state = net.forward(next_state.clone())
        I *= gamma
        step_data.append([state, action, reward, policy, value, value_next_state, I])
        state = next_state
        ep_return += reward
    data = DataLoader(step_data, 100, shuffle=False)
    if episode % 5 == 0:
        iterable = iter(data)
        for e in range(10):
            print(e)
            state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts = next(iterable)
            net.zero_grad()
            target = reward_t + gamma * value_next_state_t
            critic_loss = F.mse_loss(value_t.squeeze(), target)
            advantage = (target - value_t).squeeze()
            new_probs, _ = net.forward(state_t.clone())
            new_log_probs = torch.log(new_probs.squeeze() + 1e-6)
            log_probs = torch.log(policy_t.squeeze() + 1e-6)
            action_log_probs = log_probs.gather(1, action_t.squeeze().unsqueeze(1))
            new_action_log_probs = new_log_probs.gather(1, action_t.squeeze().unsqueeze(1))
            prob_ratio = torch.exp(new_action_log_probs)/torch.exp(action_log_probs)
            weighted_probs = prob_ratio * advantage
            weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip, 1+clip)*advantage
            entropy = -torch.sum(policy_t.squeeze() * log_probs.squeeze(), dim = -1, keepdim = True)
            actor_loss = torch.sum(- discounts * torch.minimum(weighted_probs, weighted_clipped_probs) - 0.02*entropy)
            total_loss = actor_loss + critic_loss
            total_loss.backward()
            # optimizer.step()
        step_data = []