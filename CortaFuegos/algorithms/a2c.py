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
import copy
# A2C algorithm:
def actor_critic(env, net, episodes, version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.01):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            policy, value = net.forward(copy.copy(state))
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            _, value_next_state = net.forward(next_state)
            target = reward + gamma * value_next_state
            net.zero_grad()
            critic_loss = F.mse_loss(value, target)
            critic_loss.backward(retain_graph=True)
            advantage = (target - value)
            log_probs = torch.log(policy + 1e-6)
            # print(log_probs.shape)
            # print(action.shape)
            action_log_probs = log_probs.gather(1, action)
            entropy = -torch.sum(policy * log_probs, dim = -1, keepdim = True)
            actor_loss = -I * action_log_probs * advantage - beta*entropy
            actor_loss.backward()
            optimizer.step()
            ep_return += reward
            state = next_state
            I *= gamma
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())
        if episode in plot_episode:
            plot_prog(env, episode, net, version ,"figures", "a2c" )
    return stats

