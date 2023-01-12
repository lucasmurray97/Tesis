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
from torch.optim.lr_scheduler import LambdaLR
import json
import copy
import random
def dqn2(env, net, episodes, env_version, net_version, plot_episode, alpha = 1e-5, gamma = 0.99, beta = 0.02, landa = 0.95, epsilon = 1, n_envs = 8, epochs = 10, batch_size = 64, instance = "sub20x20", test = False, window = 10, demonstrate = True, n_dem = 10, combined = False, temporal = False, prioritized = False, max_mem = 1000, target_update = 1, epsilon_dec = 0.01, epsilon_min = 0.005, lr_decay = 0.01):
    optimizer = AdamW(net.parameters(), lr = alpha)
    lambda1 = lambda epoch: 1/(1 + lr_decay*epoch)
    scheduler = LambdaLR(optimizer,lr_lambda=lambda1)
    env_shape = env.env_shape
    memory = ReplayMemory(env_shape, max_mem=max_mem, batch_size=batch_size, demonstrate=demonstrate, n_dem=n_dem, combined=combined, temporal=temporal, env="FG", version=env_version[1], size=env_shape[1],n_envs=n_envs, gamma = gamma, landa = landa, prioritized=prioritized)
    if prioritized:
        states, actions, rewards, next_states, gammas, landas, dones = memory.buffer.get_all()
        q_pred = net.forward(states).gather(1, actions.unsqueeze(1).type(torch.int64))
        q_target = net.max(net.forward(next_states), next_states)
        target = rewards + gammas*q_target*(~dones)
        errors = target - q_pred.squeeze(1)
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
            q = net.forward(state_c)
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
                q_target = net.max(net.forward(next_state), next_state)
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
                q_pred = net.forward(state_t).gather(1, action_t.unsqueeze(1).type(torch.int64))
                next_q_pred = net.forward(next_state_t)
                max_action = net.sample(next_q_pred, next_state_t)
                q_target = target_net.forward(next_state_t).gather(1, max_action.type(torch.int64)).squeeze(1)
                target = reward_t + gamma*q_target
                criterion = nn.SmoothL1Loss()
                if prioritized:
                    errors = target - q_pred
                    memory.buffer.set_priority(indices, errors)
                    total_loss = criterion(torch.sum(q_pred*(importance**(1-epsilon))), torch.sum(target*(importance**(1-epsilon))))
                else:
                    total_loss = criterion(torch.sum(q_pred), torch.sum(target))
                total_loss.backward()
                torch.nn.utils.clip_grad_value_(net.parameters(),100.)
                optimizer.step()
            stats["Loss"].append(total_loss.detach().mean().item())
            curr_lr = optimizer.param_groups[0]['lr']
            if curr_lr > 1e-10:
                scheduler.step()
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
        #     plot_prog(env.envs[0], episode, net, env_version, net_version ,"2dqn", env.size, instance, test)
    params = {"alpha": alpha, "gamma": gamma, "epsilon": epsilon, "target_update": target_update}
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "2dqn", window = window, instance = instance, test = test, params = params)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "2dqn", test)
    plot_trayectory_probs(env.envs[0], episode, net, env_version, net_version ,"2dqn", env.size, instance, test, params = params)
    params_dir = f"episodes={episodes*n_envs}_"
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    with open(f"data/{env.envs[0].name}/{instance}/{env_version}/{net_version}/2dqn/stats_{params_dir}.json", "w+") as write_file:
        json.dump(stats, write_file, indent=4)
    return stats