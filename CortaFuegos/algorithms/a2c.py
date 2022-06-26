import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.parallel_firegrid import Parallel_Firegrid
import torch.nn.functional as F


# A2C algorithm:
def actor_critic(env, policy, value_net, episodes, alpha = 1e-4, gamma = 0.99, beta = 0.01):
    policy_optim = AdamW(policy.parameters(), lr = alpha)
    value_net_optim = AdamW(value_net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        while not done:
            action = policy(state).multinomial(1).detach()
            next_state, reward, done = env.step(action)
            value = value_net(state)
            target = reward + gamma * value_net(next_state).detach()
            critic_loss = F.mse_loss(value, target)
            value_net.zero_grad()
            critic_loss.backward()
            value_net_optim.step()
            
            advantage = (target - value).detach()
            probs = policy(state)
            log_probs = torch.log(probs + 1e-6)
            action_log_probs = log_probs.gather(0, action)
            entropy = -torch.sum(probs * log_probs, dim = -1, keepdim = True)
            actor_loss = -I * action_log_probs * advantage - beta*entropy
            actor_loss = actor_loss.mean()
            policy.zero_grad()
            actor_loss.backward()
            policy_optim.step()
            
            ep_return += reward
            state = next_state
            I *= gamma
        stats["Actor Loss"].append(actor_loss.item())
        stats["Critic Loss"].append(critic_loss.item())
        stats["Returns"].append(ep_return.mean().item())
    return stats

