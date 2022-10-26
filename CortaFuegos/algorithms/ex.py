from cnn_policy_value_v2_2 import CNN as CNN_2
from cnn_policy_value_v3_2 import CNN as CNN_3
from enviroment.firegrid_v5 import FireGrid_V5
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v4 import FireGrid_V4
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import numpy as np
from ppo_coupled import ppo
from q_learning_2 import q_learning
# net = CNN_3(input_size = 4)
n_envs = 8
env = Parallel_Wrapper(FireGrid_V6, n_envs = n_envs, parameters = {"size": 2, "burn_value": 10, "n_sims" : 10})
episodes = 10
returns, q_table = q_learning(2, env, episodes, "v6", plot_episode = [], alpha = 1e-5, gamma = 0.99, n_envs = 8, epsilon = 0.15)
print(returns)
