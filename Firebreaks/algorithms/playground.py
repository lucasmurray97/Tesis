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
from nets.local_small_net_q_v2 import LOCAL_CNN_SMALL_Q_v2
from nets.mobile_net import small_mobile, big_mobile
from nets.efficient_net import efficient_net
from algorithms.ddqnet import ddqnet
net = efficient_net(20, 1, 400, only_q=False, version=2)
net.eval()
env = Full_Grid_V1(size = 20)
env_shape = (1, 20, 20)
state = env.reset()
q, _ = net(state)
print(sum(p.numel() for p in net.parameters() if p.requires_grad))