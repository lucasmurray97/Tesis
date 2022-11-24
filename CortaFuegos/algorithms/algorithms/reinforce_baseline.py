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
torch.autograd.set_detect_anomaly(True)
# reinforce with baseline algorithm:
def reinforce_baseline(env, net, episodes, env_version, net_version, plot_episode, n_envs = 8, alpha = 1e-4, gamma = 0.99, beta = 0.02, instance = "sub20x20", test = False, window = 10):
    optimizer = AdamW(net.parameters(), lr = alpha)
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
        ep_values = torch.zeros((ep_len, n_envs, 1))
        ep_next_values = torch.zeros((ep_len, n_envs, 1))
        ep_I = torch.zeros((ep_len, n_envs, 1))
        while not done:
            state_c = state.clone()
            mask = env.generate_mask()
            policy, value, entropy = net.forward(state_c, mask = mask)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            next_mask = env.generate_mask()
            _, value_next_state, _ = net.forward(next_state_c, mask = next_mask)
            ep_actions[step] = action.clone()
            ep_rewards[step] = reward.clone()
            ep_policy[step] = policy.clone()
            ep_masks[step] = mask
            ep_entropy[step] = entropy.unsqueeze(1).clone().detach()
            ep_values[step] = value.clone().detach()
            ep_next_values[step] = value_next_state.clone().detach()
            ep_I[step] = torch.full((n_envs, 1), I)
            step = step + 1
            I *= gamma
            state = next_state
            ep_return += reward
        action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts = torch.transpose(ep_actions, 0, 1), torch.transpose(ep_rewards, 0, 1), torch.transpose(ep_policy, 0, 1), torch.transpose(ep_entropy, 0, 1), torch.transpose(ep_masks, 0, 1), torch.transpose(ep_values, 0, 1), torch.transpose(ep_next_values, 0, 1), torch.transpose(ep_I, 0, 1)
        data = DataLoader([[action_t[i], reward_t[i], policy_t[i], entropy_t[i], mask_t[i], value_t[i], value_next_state_t[i], discounts[i]] for i in range(n_envs)], n_envs, shuffle = False)
        actor_loss_acum = 0
        critic_loss_acum = 0
        total_loss_acum = 0
        for action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts in data:
            action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts= map(lambda i: torch.flatten(i, end_dim = 1), [action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts])
            net.zero_grad()
            target = reward_t + gamma * value_next_state_t
            critic_loss = F.mse_loss(value_t, target)
            log_probs = torch.log(policy_t + 1e-6)
            action_log_probs = log_probs.gather(1, action_t)
            entropy = entropy_t
            discounted_rewards = discounts.flip(0) * reward_t
            G = torch.cumsum(discounted_rewards, dim=0)
            actor_loss = torch.sum(- (G - value_t) * action_log_probs * discounts - beta*entropy)
            total_loss = critic_loss + actor_loss
            critic_loss_acum += critic_loss
            actor_loss_acum += actor_loss
            total_loss_acum = critic_loss_acum + actor_loss_acum
            total_loss.backward()
            optimizer.step()
        stats["Loss"].append(total_loss_acum.detach().mean())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version, "reinforce_baseline", env.size, instance = instance, test = test)
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "reinforce_baseline", window = window, instance = instance, test = test)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "reinforce_baseline", test)
    return stats

