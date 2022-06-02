import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from enviroment.parallel_firegrid import Parallel_Firegrid

# Reinforce Algorithm:
def reinforce(env, policy, episodes, alpha = 1e-4, gamma = 0.99):
    optim = AdamW(policy.parameters(), lr = alpha)
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        transitions = []
        ep_return = 0
        while not done:
            action = policy(state).multinomial(1).detach()
            next_state, reward, done = env.step(action)
            transitions.append([state, action, reward])
            ep_return += reward
            state = next_state
        G = torch.zeros((12, 1))
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma * G
            probs_t = policy(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(1, action_t)
            entropy_t = -torch.sum(probs_t * log_probs_t, dim = -1, keepdim = True)
            gamma_t = gamma**t
            pg_loss_t = -gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - 0.01*entropy_t).mean()
            policy.zero_grad()
            total_loss_t.backward()
            optim.step()
        stats["Loss"].append(total_loss_t.item())
        stats["Returns"].append(ep_return.mean().item())
    return stats





