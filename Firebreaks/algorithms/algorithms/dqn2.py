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
from algorithms.utils.annealing import LinearSchedule
import json
import copy
import random
import os
def dqn2(env, net, episodes, env_version, net_version, alpha = 1e-5, gamma = 0.99, exploration_fraction = 0.3, landa = 0.95, epsilon = 1, n_envs = 8, pre_epochs = 1000, batch_size = 64, instance = "sub20x20", test = False, window = 10, demonstrate = True, n_dem = 10, prioritized = False, max_mem = 1000, target_update = 100, lr_decay = 0.01, lambda_1=1.0, lambda_2=1.0, gpu=False):
    total_timesteps = env.envs[0].get_episode_len()*episodes
    optimizer = AdamW(net.parameters(), lr = alpha)
    lambda1 = lambda epoch: 1/(1 + lr_decay*epoch)
    scheduler = LambdaLR(optimizer,lr_lambda=lambda1)
    env_shape = env.env_shape
    memory = ReplayMemory(env_shape, max_mem=max_mem, batch_size=batch_size, demonstrate=demonstrate, n_dem=n_dem, env="FG", version=env_version[1], size=env_shape[1],n_envs=n_envs, gamma = gamma, landa = landa, prioritized=prioritized, instance=instance, gpu=gpu)
    target_net = copy.deepcopy(net)
    exploration = LinearSchedule(schedule_timesteps=int(exploration_fraction * total_timesteps),
                                 initial_p=epsilon,
                                 final_p=0.02)
    if prioritized:
        beta_schedule = LinearSchedule(total_timesteps,
                                       initial_p=0.4,
                                       final_p=1.0)
    initial_epsilon = epsilon
    if demonstrate:
        print("Pre-Training started!")
        updates = 0
        for _ in tqdm(range(pre_epochs)):
            indices, state_t, action_t, reward_t, next_state_t, _, _, done_t, importance, dem = memory.buffer.sample_memory()
            net.zero_grad()
            q_pred_e = net.forward(state_t).gather(1, action_t.unsqueeze(1).type(torch.int64))
            q_target = target_net.forward(state_t)
            q_pred_next = net.forward(next_state_t)
            max_action_next = net.sample(q_pred_next, next_state_t)
            q_target_next = target_net.forward(next_state_t).gather(1, max_action_next.type(torch.int64)).squeeze(1)
            J_E = target_net.je_loss(action_t, q_target, state_t, dem) - torch.sum(q_pred_e)
            target = reward_t + gamma*q_target_next*(~done_t)
            criterion = nn.SmoothL1Loss()
            if prioritized:
                J_DQN = criterion(torch.sum(q_pred_e*(importance**(1-epsilon))), torch.sum(target*(importance**(1-epsilon))))
            else:
                J_DQN = criterion(torch.sum(q_pred_e), torch.sum(target))
            n_rewards, n_state, use = memory.buffer.get_n_steps(indices)
            n_q_target = target_net.forward(n_state)
            n_max_action = net.sample(n_q_target, n_state)
            n_target = n_rewards[:,0] + n_rewards[:,1] * gamma + (gamma**2)*n_q_target.gather(1, n_max_action.type(torch.int64)).squeeze(0)*(use.squeeze(1))
            J_N = criterion(torch.sum(q_pred_e), torch.sum(n_target))
            J = J_DQN + lambda_1*J_N + lambda_2*J_E
            J.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(),100.)
            optimizer.step()
            updates += 1
            if prioritized:
                errors = target - q_pred_e.squeeze(1)
                memory.buffer.set_priority(indices, errors)
            if updates % target_update == 0:
                target_net.load_state_dict(net.state_dict())
        print("Finished Pre-Training!")
    stats = {"Loss": [], "Returns": []}
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
            ep_return += reward
            discounts *=gamma
            I *=landa
            memory.buffer.store_transition(state, action, reward, next_state, done, discounts, I)
            if steps % target_update == 0:
                target_net.load_state_dict(net.state_dict())
            epsilon = exploration.value(steps)
            state = next_state
            step = step + 1
            steps = steps + 1
        if memory.is_sufficient():
            if prioritized:
                    memory.buffer.set_beta(value=beta_schedule.value(steps))
            indices, state_t, action_t, reward_t, next_state_t, _, _, done_t, importance, dem = memory.buffer.sample_memory()
            net.zero_grad()
            q_pred = net.forward(state_t).gather(1, action_t.unsqueeze(1).type(torch.int64))
            q_target = target_net.forward(state_t)
            q_pred_next = net.forward(next_state_t)
            max_action_next = net.sample(q_pred_next, next_state_t)
            q_target_next = target_net.forward(state_t).gather(1, max_action_next.type(torch.int64)).squeeze(1)
            J_E = target_net.je_loss(action_t, q_target, state_t, dem) - torch.sum(q_pred*dem)
            target = reward_t + gamma*q_target_next*(~done_t)
            criterion = nn.SmoothL1Loss()
            if prioritized:
                J_DQN = criterion(torch.sum(q_pred*importance), torch.sum(target*importance))
            else:
                J_DQN = criterion(torch.sum(q_pred), torch.sum(target))
            n_rewards, n_state, use = memory.buffer.get_n_steps(indices)
            n_q_target = target_net.forward(n_state)
            n_max_action = net.sample(n_q_target, n_state)
            n_target = n_rewards[:,0] + n_rewards[:,1] * gamma + (gamma**2)*n_q_target.gather(1, n_max_action.type(torch.int64)).squeeze(0)*(use.squeeze(1))
            J_N = criterion(torch.sum(q_pred), torch.sum(n_target))
            J = J_DQN + lambda_1*J_N + lambda_2*J_E
            J.backward()
            torch.nn.utils.clip_grad_value_(net.parameters(),100.)
            optimizer.step()
            if prioritized:
                errors = target - q_pred.squeeze(1)
                memory.buffer.set_priority(indices, errors)
            stats["Loss"].append(J.detach().mean().item())
            curr_lr = optimizer.param_groups[0]['lr']
            if curr_lr > 1e-10:
                scheduler.step()
        if n_envs != 1:
            stats["Returns"].extend(ep_return.squeeze().tolist())
        else:
            stats["Returns"].append(ep_return)
    params = {"alpha": alpha, "gamma": gamma, "epsilon": initial_epsilon, "target_update": target_update, "prioritized": prioritized, "n_dem": n_dem, "exploration": exploration_fraction, 'pre_epochs': pre_epochs}
    plot_moving_av(env.envs[0], stats["Returns"], episodes*n_envs, env_version, net_version, "2dqn", window = window, instance = instance, test = test, params = params)
    plot_loss(env.envs[0], stats["Loss"], episodes, env_version, instance, net_version, "2dqn", test, params = params)
    plot_trayectory_probs(env.envs[0], episode, net, env_version, net_version ,"2dqn", env.size, instance, test, params = params)
    params_dir = f"episodes={episodes*n_envs}_"
    for key in params.keys():
            params_dir += key + "=" + str(params[key]) + "_"
    try:
        os.makedirs(f"data/{env.envs[0].name}/{env_version}/{instance}/sub{env.envs[0].size}x{env.envs[0].size}/{net_version}/2dqn/")
    except OSError as error:  
        print(error)
    with open(f"data/{env.envs[0].name}/{env_version}/{instance}/sub{env.envs[0].size}x{env.envs[0].size}/{net_version}/2dqn/stats_{params_dir}.json", "w+") as write_file:
        json.dump(stats, write_file, indent=4)
    return stats