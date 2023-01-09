from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2
from algorithms.utils.q_learning_tools import Q_Table
from algorithms.mab_ucb import MAB_UCB_FG
import torch
from nets.small_net_v1 import CNN_SMALL_V1
from nets.small_net_v2 import CNN_SMALL_V2
from nets.small_net_q import CNN_SMALL_Q
from nets.big_net_q import CNN_BIG_Q
from nets.big_net_v1 import CNN_BIG_V1
from nets.big_net_v2 import CNN_BIG_V2
from nets.mask import CategoricalMasked
import multiprocessing
from enviroment.utils.final_reward import generate_reward
from algorithms.utils.replay_buffer import ReplayMemory
from nets.mask import CategoricalMasked, generate_mask
from algorithms.ddqnet import ddqnet
net = CNN_SMALL_Q(20, 1, 400, only_q=True, version=2).cuda()
env = Parallel_Wrapper(Full_Grid_V1, n_envs=16, parameters={'size': 20})
env_shape = env.env_shape
memory = ReplayMemory(env_shape, max_mem=10000, batch_size=64, demonstrate=True, n_dem=100, combined=False, temporal=False, prioritized=True, env="FG", version=1, size=env_shape[1],n_envs=16, gamma = 1., landa = 1.)
states, actions, rewards, next_states, gammas, landas, dones = memory.buffer.get_all()
q_pred = net.forward(states).gather(1, actions.unsqueeze(1).type(torch.int64))
print(q_pred.shape)
q_target = net.max(net.forward(next_states), next_states)
print(q_target.shape)
target = rewards + gammas*q_target*(~dones)
print(target.shape)
errors = target - q_pred.squeeze(1)
print(errors.shape)
memory.buffer.set_example_priorities(errors)
memory.buffer.sample_memory()
probs = memory.buffer.get_probs()
print(memory.buffer.get_importance(probs))



