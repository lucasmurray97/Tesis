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

def ppo_v2(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.02, clip = 0.2, landa = 0.95, n_envs = 8, epochs = 10, instance = "sub20x20", test = False, window = 10):
    ep_len = env.envs[0].get_episode_len()
    if ep_len == 1:
        ep_len = 2
    env_shape = env.env_shape
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Loss": [], "Returns": []}
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        epp_actor_loss  = 0
        epp_critic_loss  = 0
        I = 1.
        D = 1.
        step = 0
        ep_states = torch.zeros((ep_len//2, n_envs , env_shape[0] - 1, env_shape[1], env_shape[2]))
        ep_actions = torch.zeros((ep_len//2, n_envs, 1), dtype = torch.int64)
        ep_rewards = torch.zeros((ep_len//2, n_envs, 1))
        ep_policy = torch.zeros((ep_len//2, n_envs, env.envs[0].get_action_space().shape[0]))
        ep_masks = torch.zeros((ep_len//2, n_envs, env.envs[0].get_action_space().shape[0]), dtype = torch.bool)
        ep_entropy = torch.zeros((ep_len//2, n_envs, 1))
        ep_values = torch.zeros((ep_len//2, n_envs, 1))
        ep_next_values = torch.zeros((ep_len//2, n_envs, 1))
        ep_I = torch.zeros((ep_len//2, n_envs, 1))
        ep_D = torch.zeros((ep_len//2, n_envs, 1))
        while not done:
            state_c = state.clone()
            mask = env.generate_mask()
            policy, value, entropy = net.forward(state_c, mask = mask)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            next_mask = env.generate_mask()
            _, value_next_state, _ = net.forward(next_state_c, mask = next_mask)
            I = I * gamma
            D = D * landa
            ep_states[step % (ep_len//2)] = state_c
            ep_actions[step % (ep_len//2)] = action.clone()
            ep_rewards[step % (ep_len//2)] = reward.clone()
            ep_policy[step % (ep_len//2)] = policy.clone().detach()
            ep_masks[step % (ep_len//2)] = mask
            ep_entropy[step % (ep_len//2)] = entropy.unsqueeze(1).clone().detach()
            ep_values[step % (ep_len//2)] = value.clone().detach()
            ep_next_values[step % (ep_len//2)] = value_next_state.clone().detach()
            ep_I[step % (ep_len//2)] = torch.full((n_envs, 1), I)
            ep_D[step % (ep_len//2)] = torch.full((n_envs, 1), D)
            if step % (ep_len//2) == 0 and step != 0:
                state_t, action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts, landas = torch.transpose(ep_states, 0, 1), torch.transpose(ep_actions, 0, 1), torch.transpose(ep_rewards, 0, 1), torch.transpose(ep_policy, 0, 1), torch.transpose(ep_entropy, 0, 1), torch.transpose(ep_masks, 0, 1), torch.transpose(ep_values, 0, 1), torch.transpose(ep_next_values, 0, 1), torch.transpose(ep_I, 0, 1), torch.transpose(ep_D, 0, 1)
                data = DataLoader([[state_t[i], action_t[i], reward_t[i], policy_t[i], entropy_t[i], mask_t[i] , value_t[i], value_next_state_t[i], discounts[i], landas[i]] for i in range(n_envs)], 1, shuffle = False)
                loss_acum = 0
                for e in range(epochs):
                    n = 0
                    for state_t, action_t, reward_t, policy_t, entropy_t, mask_t, value_t, value_next_state_t, discounts, landas in data:
                        net.zero_grad()
                        target = reward_t.squeeze(0) + gamma * value_next_state_t.squeeze(0)
                        critic_loss = F.mse_loss(value_t.squeeze(0), target)
                        delta = (reward_t.squeeze(0) + landa*value_next_state_t.squeeze(0) - value_t.squeeze(0)).squeeze()
                        advantage = torch.cumsum(delta * I * landas.squeeze(0), dim=0)
                        new_probs, _, _ = net.forward(state_t.squeeze(0).clone(), mask_t.squeeze(0))
                        new_log_probs = torch.log(new_probs + 1e-6)
                        log_probs = torch.log(policy_t.squeeze(0) + 1e-6)
                        action_log_probs = log_probs.gather(1, action_t.squeeze(0))
                        new_action_log_probs = new_log_probs.gather(1, action_t.squeeze(0))
                        prob_ratio = torch.exp(new_action_log_probs) - torch.exp(action_log_probs)
                        weighted_probs = prob_ratio * advantage
                        weighted_clipped_probs = torch.clamp(prob_ratio, 1 - clip, 1 + clip)*advantage
                        entropy = entropy_t
                        clip_loss = torch.sum(discounts.squeeze(0) * torch.minimum(weighted_probs, weighted_clipped_probs))
                        total_loss = critic_loss - (clip_loss + torch.sum(beta*entropy))
                        loss_acum += total_loss
                        total_loss.backward()
                        optimizer.step()
                        n+=1
                stats["Loss"].append(loss_acum.detach().mean())
            state = next_state
            ep_return += reward
            step = step + 1 
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version ,"ppo_v2", env.size, instance = instance, test = test)
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "ppo_v2", window = window, instance = instance, test = test)   
    plot_loss(env.envs[0], stats["Loss"], episodes*2, env_version, instance, net_version, "ppo_v2", test)
    return stats