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


def ppo_v2(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-4, gamma = 0.99, beta = 0.02, clip = 0.2, landa = 0.95, update_step = 50):
    optimizer = AdamW(net.parameters(), lr = alpha)
    stats = {"Actor Loss": [], "Critic Loss": [], "Returns": []}
    step_data = []
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        epp_actor_loss  = 0
        epp_critic_loss  = 0
        I = 1.
        D = 1.
        step = 1
        while not done:
            state_c = state.clone()
            policy, value = net.forward(state_c.unsqueeze(0))
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            next_state_c = next_state.clone()
            _, value_next_state = net.forward(next_state_c.unsqueeze(0))
            I = I * gamma
            D = D * landa
            step_data.append([state_c, action.clone(), reward.clone(), policy.clone().detach(), value.clone().detach(), value_next_state.clone().detach()*done, I, D])
            if step % update_step == 0:
                data = DataLoader(step_data, 10, shuffle=False)
                for e in range(5):
                    for state_t, action_t, reward_t, policy_t, value_t, value_next_state_t, discounts, D in data:
                        net.zero_grad()
                        delta = (reward_t + landa*value_next_state_t - value_t).squeeze()
                        advantage = torch.cumsum(delta * I * D, dim=0)
                        target = reward_t + gamma * value_next_state_t
                        critic_loss = F.mse_loss(value_t, target)
                        new_probs, _ = net.forward(state_t.clone())
                        new_log_probs = torch.log(new_probs + 1e-6)
                        log_probs = torch.log(policy_t + 1e-6)
                        action_log_probs = log_probs.gather(2, action_t).squeeze(1)
                        new_action_log_probs = new_log_probs.gather(1, action_t.squeeze(1))
                        prob_ratio = torch.exp(new_action_log_probs)/torch.exp(action_log_probs)
                        weighted_probs = prob_ratio * advantage
                        weighted_clipped_probs = torch.clamp(prob_ratio, 1-clip, 1+clip)*advantage
                        entropy = -torch.sum(policy_t * log_probs, dim = -1, keepdim = True).squeeze(1)
                        actor_loss = torch.sum(-discounts * torch.minimum(weighted_probs, weighted_clipped_probs) - beta*entropy)
                        total_loss = actor_loss + critic_loss
                        total_loss.backward()
                        optimizer.step()
                        epp_actor_loss = epp_actor_loss + actor_loss
                        epp_critic_loss = epp_critic_loss + critic_loss
                step_data = []
            state = next_state
            ep_return += reward
            step = step + 1 
        if episode in plot_episode:
            plot_prog(env, episode, net, env_version, net_version ,"figures", "ppo_v2" )
        stats["Returns"].append(ep_return.mean().item())
        stats["Actor Loss"].append(epp_actor_loss.mean().item())
        stats["Critic Loss"].append(epp_critic_loss.mean().item())
    return stats