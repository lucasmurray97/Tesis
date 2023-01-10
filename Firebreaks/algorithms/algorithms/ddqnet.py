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
import random
def ddqnet(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-5, gamma = 0.99, beta = 0.02, landa = 0.95, epsilon = 1, n_envs = 8, epochs = 10, batch_size = 64, instance = "sub20x20", test = False, window = 10, demonstrate = True, n_dem = 10, combined = False, temporal = False, prioritized = False, max_mem = 1000, target_update = 1, epsilon_dec = 0.01, epsilon_min = 0.005):
    optimizer = AdamW(net.parameters(), lr = alpha)
    env_shape = env.env_shape
    ep_len = env.envs[0].get_episode_len()
    memory = ReplayMemory(env_shape, max_mem=max_mem, batch_size=batch_size, demonstrate=demonstrate, n_dem=n_dem, combined=combined, temporal=temporal, prioritized=prioritized, env="FG", version=env_version[1], size=env_shape[1],n_envs=n_envs, gamma = gamma, landa = landa)
    if prioritized:
        states, actions, rewards, next_states, gammas, landas, dones = memory.buffer.get_all()
        adv, v = net.forward(states)
        q_pred = (v + (adv - adv.mean(dim=1, keepdim=True))).gather(1, actions.unsqueeze(1).type(torch.int64)).squeeze(1)
        target_advantage, target_value = net.forward(next_states)
        q_target = net.max((target_value + (target_advantage - target_advantage.mean(dim=1, keepdim=True))), next_states)
        target = rewards + gammas*q_target*(~dones)
        errors = target - q_pred
        memory.buffer.set_example_priorities(errors) 
    stats = {"Loss": [], "Returns": []}
    target_net = copy.deepcopy(net)
    steps = 0
    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return  = 0
        step = 0
        discounts = 1.
        I = 1.
        while not done:
            state_c = state.clone()
            adv, v = net.forward(state_c)
            q = (v + (adv - adv.mean(dim=1, keepdim=True)))
            if random.uniform(0, 1) > epsilon:
                action = net.sample(q, state_c)
            else:
                action = env.random_action()
            next_state, reward, done = env.step(action.detach())
            state = next_state
            ep_return += reward
            discounts *=gamma
            I *=landa
            if prioritized:
                target_advantage, target_value = target_net.forward(next_state)
                q_target = net.max((target_value + (target_advantage - target_advantage.mean(dim=1, keepdim=True))), next_state)
                target = reward.squeeze(1) + discounts*q_target*(~done)
                error = target - q.gather(1, action.type(torch.int64)).squeeze(1)
                memory.buffer.store_transition(state, action, reward, next_state, error, step, done, discounts, I)
            else:
                memory.buffer.store_transition(state, action, reward, next_state, step, done, discounts, I)
            if steps % target_update == 0:
                target_net.load_state_dict(net.state_dict())
            step = step + 1
            steps = steps + 1
        if memory.is_sufficient():
            indices, state_t, action_t, reward_t, next_state_t, _, _, done_t, importance = memory.buffer.sample_memory()
            for e in range(epochs):
                net.zero_grad()
                advantage_t, value_t = net.forward(state_t)
                next_advantage_t, next_value_t = net.forward(next_state_t)
                target_advantage_t, target_value_t = target_net.forward(next_state_t)
                q_pred = (value_t + (advantage_t - advantage_t.mean(dim=1, keepdim=True)) ).gather(1, action_t.unsqueeze(1).type(torch.int64)).squeeze(1)
                next_q_pred = (next_value_t + (next_advantage_t - next_advantage_t.mean(dim=1, keepdim=True)))
                max_action = net.sample(next_q_pred, next_state_t)
                target_q_pred = (target_value_t + (target_advantage_t - target_advantage_t.mean(dim=1, keepdim=True))).gather(1, max_action.type(torch.int64)).squeeze(1)
                target = reward_t + gamma*target_q_pred*(~done_t)
                if prioritized:
                    errors = target - q_pred
                    memory.buffer.set_priority(indices, errors.unsqueeze(1))
                    total_loss = F.mse_loss(torch.sum(q_pred*(importance**(1-epsilon))), torch.sum(target*(importance**(1-epsilon))))
                else:
                    total_loss = F.mse_loss(torch.sum(q_pred), torch.sum(target))
                total_loss.backward()
                optimizer.step()
            stats["Loss"].append(total_loss.detach().mean().item())
        if episode % target_update == 0:
            target_net.load_state_dict(net.state_dict())
        if epsilon > epsilon_min:
            epsilon = epsilon - epsilon_dec
        else:
            epsilon = epsilon_min
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
        # if episode in plot_episode:
        #     plot_prog(env.envs[0], episode, net, env_version, net_version ,"ddqn", env.size, instance, test)
    params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "target_update": target_update}
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "ddqn", window = window, instance = instance, test = test, params = params)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "ddqn", test)
    plot_trayectory_probs(env.envs[0], episode, net, env_version, net_version ,"ddqn", env.size, instance, test, params = params)
    params_dir = f"episodes={episodes*n_envs}_"
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    with open(f"data/{env.envs[0].name}/{instance}/{env_version}/{net_version}/ddqn/stats_{params_dir}.json", "w+") as write_file:
        json.dump(stats, write_file, indent=4)
    return stats