import os
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
from algorithms.utils.plot_progress import plot_prog
from torch.utils.data import Dataset, DataLoader
from algorithms.utils.plot_progress import plot_moving_av, plot_loss
# Reinforce Algorithm:
def reinforce(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.1, n_envs = 8, instance = "sub20x20", test = False, window = 10):
    optim = AdamW(net.parameters(), lr = alpha)
    env_shape = env.env_shape
    ep_len = env.envs[0].get_episode_len()
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        step = 0
        ep_actions = torch.zeros((ep_len, n_envs, 1), dtype = torch.int64)
        ep_rewards = torch.zeros((ep_len, n_envs, 1))
        ep_policy = torch.zeros((ep_len, n_envs, env.envs[0].get_action_space().shape[0]))
        ep_masks = torch.zeros((ep_len, n_envs, env.envs[0].get_action_space().shape[0]), dtype = torch.bool)
        ep_entropy = torch.zeros((ep_len, n_envs, 1))
        ep_I = torch.zeros((ep_len, n_envs, 1))
        while not done:
            state_c = state.clone()
            mask = env.generate_mask()
            policy, entropy = net.forward(state_c, mask = mask)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            ep_return += reward
            state = next_state
            ep_actions[step] = action.clone()
            ep_rewards[step] = reward.clone()
            ep_policy[step] = policy.clone()
            ep_masks[step] = mask
            ep_entropy[step] = entropy.unsqueeze(1).clone().detach()
            ep_I[step] = torch.full((n_envs, 1), I)
            step = step + 1
            I *= gamma
            state = next_state
            ep_return += reward
        action_t, reward_t, policy_t, entropy_t, mask_t, discounts = torch.transpose(ep_actions, 0, 1), torch.transpose(ep_rewards, 0, 1), torch.transpose(ep_policy, 0, 1), torch.transpose(ep_entropy, 0, 1), torch.transpose(ep_masks, 0, 1), torch.transpose(ep_I, 0, 1)
        data = DataLoader([[action_t[i], reward_t[i], policy_t[i], entropy_t[i], mask_t[i], discounts[i]] for i in range(n_envs)], n_envs, shuffle = False)
        loss_acum = 0
        for action_t, reward_t, policy_t, entropy_t, mask_t, discounts in data:
            action_t, reward_t, policy_t, entropy_t, mask_t, discounts= map(lambda i: torch.flatten(i, end_dim = 1), [action_t, reward_t, policy_t, entropy_t, mask_t, discounts])
            net.zero_grad()
            log_probs = torch.log(policy_t + 1e-6)
            action_log_probs = log_probs.gather(1, action_t)
            entropy = entropy_t
            discounted_rewards = discounts.flip(0) * reward_t
            G = torch.cumsum(discounted_rewards, dim=0)
            total_loss = torch.sum(- (G * action_log_probs * discounts) - beta*entropy)
            loss_acum += total_loss
            total_loss.backward()
            optim.step()
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version, "reinforce", env.size, instance = instance, test = test)
        stats["Loss"].append(loss_acum.detach().mean())
        stats["Returns"].append(ep_return.mean().item())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "reinforce", window = window, instance = instance, test = test)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "reinforce", test)
    return stats


