import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch import nn as nn
from torch.optim import AdamW
import numpy as np
import torch.nn.functional as F
from algorithms.utils.plot_progress import plot_prog
from torch.utils.data import Dataset, DataLoader
from algorithms.utils.plot_progress import plot_moving_av, plot_loss, plot_trayectory_probs
from algorithms.utils.replay_buffer import ReplayMemory
import json
import copy
def ppo(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-5, gamma = 0.99, beta = 0.02, clip = 0.2, landa = 0.95, n_envs = 8, epochs = 10, batch_size = 64, instance = "sub20x20", test = False, window = 10, demonstrate = True, n_dem = 10, combined = False, temporal = False, max_mem = 1000, target_update = 1):
    optimizer = AdamW(net.parameters(), lr = alpha)
    env_shape = env.env_shape
    ep_len = env.envs[0].get_episode_len()
    memory = ReplayMemory(env_shape, max_mem=max_mem, batch_size=batch_size, demonstrate=demonstrate, n_dem=n_dem, combined=combined, temporal=temporal, env="FG", version=1, size=env_shape[1],n_envs=n_envs, gamma = gamma, landa = landa)
    stats = {"Loss": [], "Returns": []}
    target_net = copy.deepcopy(net)
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        step = 0
        discounts = 1.
        I = 1.
        while not done:
            state_c = state.clone()
            policy, _, _ = target_net.forward(state_c)
            action = policy.multinomial(1)
            next_state, reward, done = env.step(action.detach())
            state = next_state
            ep_return += reward
            discounts *=gamma
            I *=landa
            memory.buffer.store_transition(state, action, reward, next_state, step, done, discounts, I)
            step = step + 1
        loss_acum = 0
        if memory.is_sufficient():
            state_t, action_t, reward_t, next_state_t, discounts_t, landas = memory.buffer.sample_memory()
            for e in range(epochs):
                net.zero_grad()
                policy_t, value_t, entropy_t = net.forward(state_t)
                _, value_next_state_t, _ = net.forward(next_state_t)
                target = reward_t.squeeze(0) + gamma * value_next_state_t.squeeze(0)
                critic_loss = F.mse_loss(value_t.squeeze(0), target)
                delta = (reward_t.squeeze(0) + landa*value_next_state_t.squeeze(0) - value_t.squeeze(0)).squeeze()
                advantage = torch.cumsum(delta * discounts_t * landas.squeeze(0), dim=0)
                new_probs, _, _ = net.forward(state_t.squeeze(0).clone())
                new_log_probs = torch.log(new_probs + 1e-6)
                log_probs = torch.log(policy_t.squeeze(0) + 1e-6)
                action_log_probs = log_probs.gather(1, action_t.unsqueeze(1).type(torch.int64))
                new_action_log_probs = new_log_probs.gather(1, action_t.unsqueeze(1).type(torch.int64))
                prob_ratio = torch.exp(new_action_log_probs) - torch.exp(action_log_probs)
                weighted_probs = prob_ratio * advantage
                weighted_clipped_probs = torch.clamp(prob_ratio, 1 - clip, 1 + clip)*advantage
                entropy = entropy_t
                clip_loss = torch.sum(discounts_t.squeeze(0) * torch.minimum(weighted_probs, weighted_clipped_probs))
                total_loss = critic_loss - (clip_loss + torch.sum(beta*entropy))
                loss_acum += total_loss
                total_loss.backward()
                optimizer.step()
        if episode % target_update:
            target_net.load_state_dict(net.state_dict)
        stats["Loss"].append(loss_acum.detach().mean().item())
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        if episode in plot_episode:
            plot_prog(env.envs[0], episode, net, env_version, net_version ,"ppo", env.size, instance, test)
    params = {"alpha": alpha, "gamma": gamma, "landa": landa, "beta": beta}
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "ppo", window = window, instance = instance, test = test, params = params)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "ppo", test)
    plot_trayectory_probs(env.envs[0], episode, net, env_version, net_version ,"ppo", env.size, instance, test, params = params)
    params_dir = f"episodes={episodes*n_envs}_"
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    with open(f"data/{env.envs[0].name}/{instance}/{env_version}/{net_version}/ppo/stats_{params_dir}.json", "w+") as write_file:
        json.dump(stats, write_file, indent=4)
    return stats