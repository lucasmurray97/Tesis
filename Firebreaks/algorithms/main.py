import sys
import torch
import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2
from algorithms.ppo_coupled import ppo
from algorithms.ppo_coupled_v2 import ppo_v2
from algorithms.ppo_coupled_v3 import ppo as ppo_v3
from algorithms.reinforce import reinforce
from algorithms.reinforce_baseline import reinforce_baseline
from algorithms.a2c import a2c
from algorithms.mab_ucb import mab_ucb
from algorithms.mab_eps_greedy import mab_greedy
from algorithms.q_learning_2 import q_learning
from algorithms.ddqnet import ddqnet
from algorithms.dqn import dqn
from algorithms.dqn2 import dqn2
from nets.small_net_v1 import CNN_SMALL_V1
from nets.big_net_q import CNN_BIG_Q
from nets.small_net_v2 import CNN_SMALL_V2
from nets.small_net_q import CNN_SMALL_Q
from nets.big_net_v1 import CNN_BIG_V1
from nets.big_net_v2 import CNN_BIG_V2
from algorithms.utils.plot_progress import plot_moving_av
from enviroment.utils.parallel_wrapper import Parallel_Wrapper 
import os
import argparse
n_envs = os.cpu_count()
# We retrieve the arguments from standard input
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, required=True)
parser.add_argument('--size', type=int, required=True)
parser.add_argument('--env', type=str, required=True)
parser.add_argument('--env_version', type=str, required=True)
parser.add_argument('--net_version', type=str, required=False)
parser.add_argument('--episodes', type=int, required=True)
parser.add_argument('--window', type=int, required=False, nargs="?", default=100)
parser.add_argument('--alpha', type=float, required=False, nargs="?")
parser.add_argument('--gamma', type=float, required=False, nargs="?", default= 1)
parser.add_argument('--landa', type=float, required=False, nargs="?", default= 1)
parser.add_argument('--exploration_fraction', type=float, required=False, nargs="?", default= 0.3)
parser.add_argument('--epsilon', type=float, required=False, nargs="?", default= 1)
parser.add_argument('--instance', type=str, required=False, nargs="?", default="homo_1")
parser.add_argument('--test', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--save_weights', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--target_update', type=int, required=False, nargs="?", default=100)
parser.add_argument('--max_mem', type=int, required=False, nargs="?", default=1000)
parser.add_argument('--demonstrate', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--n_dem', type=int, required=False, nargs="?", default=0)
parser.add_argument('--prioritized', action=argparse.BooleanOptionalAction, default=False)
parser.add_argument('--lr_decay', type=float, required=False, nargs="?", default= 0.01)
parser.add_argument('--pre_epochs', type=int, required=False, nargs="?", default = 0)
args = parser.parse_args()
# We create the enviroment
if args.env == "moving_grid":
    input_size = 2
    output_size = 16
    if args.env_version == "v6":
        env = Parallel_Wrapper(FireGrid_V6, n_envs = n_envs, parameters = {"size": args.size, "burn_value": 10, "n_sims_final" : 50})
    elif args.env_version == "v7":
        env = Parallel_Wrapper(FireGrid_V7, n_envs = n_envs, parameters = {"size": args.size, "burn_value": 10, "n_sims" : 1, "n_sims_final" : 10})
    elif args.env_version == "v8":
        env = Parallel_Wrapper(FireGrid_V8, n_envs = n_envs, parameters = {"size": args.size, "burn_value": 10, "n_sims" : 1, "n_sims_final" : 1})
    else:
        raise("Non existent version of enviroment")
elif args.env == "full_grid":
    output_size = args.size**2
    if args.env_version == "v1":
        input_size =  1
        grid_size = args.size
        env = Parallel_Wrapper(Full_Grid_V1, n_envs = n_envs, parameters = {"size": args.size, "instance": args.instance})
    elif args.env_version == "v2":
        input_size =  2
        grid_size = args.size
        env = Parallel_Wrapper(Full_Grid_V2, n_envs = n_envs, parameters = {"size": args.size, "instance": args.instance})
    else:
        raise("Non existent version of enviroment")
else:
    raise("Non existent enviroment")

start_state = env.reset()[0]
space_dims = env.envs[0].get_space_dims()[0]**2
action_dims = env.envs[0].get_action_space_dims()

if args.algorithm == "reinforce":
    value = False
else: 
    value = True
if args.test:
    path = "figures_tuning"
else:
    path = "figures"
# We create the net if needed:
if args.net_version != None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    only_q = args.algorithm == "dqn" or  args.algorithm =="2dqn"
    if args.net_version == "small":
        if args.env_version != "v2":
            if args.algorithm == "ddqn" or args.algorithm == "dqn" or  args.algorithm =="2dqn":
                net = CNN_SMALL_Q(grid_size, input_size, output_size, value, env.forbidden_cells, only_q=only_q, version = 1)
            else:
                net = CNN_SMALL_V1(grid_size, input_size, output_size, value, env.forbidden_cells)
        else:
            if args.algorithm == "ddqn" or args.algorithm == "dqn" or args.algorithm == "2dqn":
                net = CNN_SMALL_Q(grid_size, input_size, output_size, value, env.forbidden_cells, only_q=only_q, version=2)
            else:
                net = CNN_SMALL_V2(grid_size, input_size, output_size, value, env.forbidden_cells)
        net.to(device)
    elif args.net_version == "big":
        if args.env_version != "v2":
            if args.algorithm == "ddqn" or args.algorithm == "dqn" or  args.algorithm =="2dqn":
                net = CNN_BIG_Q(grid_size, input_size, output_size, value, env.forbidden_cells, only_q=only_q, version = 1)
            else:
                net = CNN_BIG_V1(grid_size, input_size, output_size, value, env.forbidden_cells)
        else:
            if args.algorithm == "ddqn" or args.algorithm == "dqn" or  args.algorithm =="2dqn":
                net = CNN_BIG_Q(grid_size, input_size, output_size, value, env.forbidden_cells, only_q=only_q, version = 2)
            else:
                net = CNN_BIG_V2(grid_size, input_size, output_size, value, env.forbidden_cells)
        net.to(device)
    else:
        raise("Non existent version of network")
    
plot_episode = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 20000, 30000, 40000, 50000]
# We retrieve the algorithm parameter:
if args.algorithm == "ppo":
    stats = ppo(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, landa = args.landa, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "ppo_v2":
    stats = ppo_v2(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, landa = args.landa, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "ppo_v3":
    stats = ppo_v3(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, landa = args.landa, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "reinforce":
    stats = reinforce(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "reinforce_baseline":
    stats = reinforce_baseline(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, combined=False, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "a2c":
    stats = a2c(env, net, args.episodes, args.env_version, args.net_version, plot_episode, alpha = args.alpha, gamma = args.gamma, beta = args.beta, instance = args.instance, test = args.test, n_envs = n_envs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update)
elif args.algorithm == "q_learning":
    returns, q_table = q_learning(args.size, env, args.episodes, args.env_version, [], alpha = args.alpha, epsilon = args.epsilon, instance = args.instance, n_envs = n_envs, window = args.window)
elif args.algorithm == "mab_ucb":
    mab_ucb(env, args.size, args.episodes, args.window, args.instance)
elif args.algorithm == "mab_greedy":
    mab_greedy(env, args.size, args.episodes, window = args.window, instance = args.instance, epsilon = args.epsilon)
elif args.algorithm == "ddqn":
    stats = ddqnet(env, net, args.episodes, args.env_version, args.net_version, alpha = args.alpha, gamma = args.gamma, landa = args.landa, exploration_fraction = args.exploration_fraction, epsilon=args.epsilon, instance = args.instance, test = args.test, n_envs = n_envs, pre_epochs = args.pre_epochs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update, prioritized=args.prioritized, lr_decay=args.lr_decay)
elif args.algorithm == "dqn":
    stats = dqn(env, net, args.episodes, args.env_version, args.net_version, alpha = args.alpha, gamma = args.gamma, landa = args.landa, exploration_fraction = args.exploration_fraction, epsilon=args.epsilon, instance = args.instance, test = args.test, n_envs = n_envs, pre_epochs = args.pre_epochs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update, prioritized=args.prioritized, lr_decay=args.lr_decay)
elif args.algorithm == "2dqn":
    stats = dqn2(env, net, args.episodes, args.env_version, args.net_version, alpha = args.alpha, gamma = args.gamma, landa = args.landa, exploration_fraction = args.exploration_fraction, epsilon=args.epsilon, instance = args.instance, test = args.test, n_envs = n_envs, pre_epochs = args.pre_epochs, window = args.window, demonstrate=args.demonstrate, n_dem=args.n_dem, max_mem=args.max_mem, target_update=args.target_update, prioritized=args.prioritized, lr_decay=args.lr_decay)
# Guardamos los parametros de la red
if args.save_weights:
    path_ = f"./weights/{args.env}/{args.instance}/{args.env_version}/{args.net_version}/{args.algorithm}.pth"
    torch.save(net.state_dict(), path_)


