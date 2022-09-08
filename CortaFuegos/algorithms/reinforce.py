import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from utils.plot_progress import plot_prog
import copy
# Reinforce Algorithm:
def reinforce(env, net, episodes, version, plot_episode, alpha = 1e-4, gamma = 0.99):
    optim = AdamW(net.parameters(), lr = alpha)
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        transitions = []
        ep_return = 0
        while not done:
            c_state = copy.copy(state)
            probs = net.forward(c_state)
            action = probs.multinomial(1).detach()
            next_state, reward, done = env.step(action)
            transitions.append([c_state, action, reward])
            ep_return += reward
            state = next_state
        G = torch.zeros(1)
        for t, (state_t, action_t, reward_t) in reversed(list(enumerate(transitions))):
            G = reward_t + gamma * G
            probs_t = net.forward(state_t)
            log_probs_t = torch.log(probs_t + 1e-6)
            action_log_prob_t = log_probs_t.gather(-1, action_t)
            entropy_t = -torch.sum(probs_t * log_probs_t, dim = -1, keepdim = True)
            gamma_t = gamma**t
            pg_loss_t = -gamma_t * action_log_prob_t * G
            total_loss_t = (pg_loss_t - 0.01*entropy_t)
            net.zero_grad()
            total_loss_t.backward()
            optim.step()
        if episode in plot_episode:
            plot_prog(env, episode, net, version ,"figures", "reinforce" )
        stats["Loss"].append(total_loss_t.item())
        stats["Returns"].append(ep_return.mean().item())
    return stats





