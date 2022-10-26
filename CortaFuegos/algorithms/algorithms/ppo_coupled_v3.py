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


def ppo(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-5, gamma = 0.99, beta = 0.02, clip = 0.2, landa = 0.95, n_envs = 8, epochs = 10, batches = 8, batch_size = 2, instance = "sub20x20", test = False):
    optimizer = AdamW(net.parameters(), lr = alpha)
    env_shape = env.env_shape
    ep_len = (env.size//2)**2
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        I = 1.
        D = 1.
        step = 0
        env_shape = env.env_shape
        ep_states = torch.zeros((ep_len, n_envs , env_shape[0] - 1, env_shape[1], env_shape[2]))
        ep_actions = torch.zeros((ep_len, n_envs, 1), dtype = torch.int64)
        ep_rewards = torch.zeros((ep_len, n_envs, 1))
        ep_policy = torch.zeros((ep_len, n_envs, 16))
        ep_values = torch.zeros((ep_len, n_envs, 1))
        ep_next_values = torch.zeros((ep_len, n_envs, 1))
        ep_I = torch.zeros((ep_len, 1))
        ep_D = torch.zeros((ep_len, 1))
        while not done:
            state_c = state.clone()
            policy, value = net.forward(state_c)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            _, value_next_state = net.forward(next_state_c)
            I = I * gamma
            D = D * landa
            state = next_state
            ep_return += reward
            ep_states[step] = state_c
            ep_actions[step] = action.clone()
            ep_rewards[step] = reward.clone()
            ep_policy[step] = policy.clone().detach()
            ep_values[step] = value.clone().detach()
            ep_next_values[step] = value_next_state.clone().detach()
            ep_I[step] = I
            ep_D[step] = D
            step = step + 1
        state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts, landas = torch.transpose(ep_states, 0, 1), torch.transpose(ep_actions, 0, 1), torch.transpose(ep_rewards, 0, 1), torch.transpose(ep_policy, 0, 1), torch.transpose(ep_values, 0, 1), torch.transpose(ep_next_values, 0, 1), torch.Tensor(ep_I), torch.Tensor(ep_D)
        data = DataLoader([[state_t[i], action_t[i], reward_t[i], policy_t[i], value_t[i], value_next_state_t[i], discounts[j], landas[j]] for i in range(n_envs) for j in range(ep_len)], batch_size, shuffle = True)
        actor_loss_acum = 0
        critic_loss_acum = 0
        n = 0
        for state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts, landas in data:
            state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts, landas = map(lambda i: torch.flatten(i, end_dim = 1), [state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts, landas])
            if n > batches:
                break
            for e in range(epochs):
                net.zero_grad()
                target = reward_t + gamma * value_next_state_t
                critic_loss = F.mse_loss(value_t, target)
                delta = (reward_t + landa*value_next_state_t - value_t)
                advantage = torch.cumsum(delta * I * landas, dim=0)
                new_probs, _ = net.forward(state_t.clone())
                new_log_probs = torch.log(new_probs + 1e-6)
                log_probs = torch.log(policy_t + 1e-6)
                action_log_probs = log_probs.gather(1, action_t)
                new_action_log_probs = new_log_probs.gather(1, action_t)
                prob_ratio = torch.exp(new_action_log_probs) - torch.exp(action_log_probs)
                weighted_probs = prob_ratio * advantage
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - clip, 1 + clip)*advantage
                entropy = -torch.sum(policy_t * log_probs, dim = -1, keepdim = True)
                actor_loss = torch.sum(-discounts * torch.minimum(weighted_probs, weighted_clipped_probs) - beta*entropy)
                total_loss = actor_loss + critic_loss
                critic_loss_acum += critic_loss
                actor_loss_acum += actor_loss
                total_loss.backward()
                optimizer.step()
            n+=1
        stats["Actor Loss"].append(actor_loss_acum.detach().mean())
        stats["Critic Loss"].append(critic_loss_acum.detach().mean())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version, "ppo_v3", env.size, instance = instance, test = test)
    return stats

