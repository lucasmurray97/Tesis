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
from nets.small_net_v2_q import CNN_SMALL_V2_Q
from nets.small_net_v1_q import CNN_SMALL_V1_Q
from nets.big_net_v1 import CNN_BIG_V1
from nets.big_net_v2 import CNN_BIG_V2
from nets.mask import CategoricalMasked
import multiprocessing
from enviroment.utils.final_reward import generate_reward
from algorithms.utils.replay_buffer import ReplayMemory
from nets.mask import CategoricalMasked, generate_mask
from algorithms.ddqnet import ddqnet
# net = CNN_SMALL_V2(6, 2, 36, True)
env = Full_Grid_V2(size=20)
state = env.reset()
net = CNN_SMALL_V2_Q(input_size=2, output_size=400, forbidden=env.forbidden_cells, only_q=True)
net.to('cuda')
# stats = ddqnet(env, net, 1, 1, "small", []) 
# print(stats)
# memory = ReplayMemory(env.env_shape, max_mem=1000, batch_size=64, temporal=False, demonstrate=True, n_dem=10)
# print(memory.buffer.sample_memory()[1])
for i in range(20):
    state = env.reset()
    done = False
    step = 0
    while not done:
        a = env.random_action()
        s, r, done = env.step(a)
        v = net.forward(s.cuda())
        print(v.shape)
        step += 1



