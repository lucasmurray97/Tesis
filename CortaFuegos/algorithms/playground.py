from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
from enviroment.full_grid_v1 import Full_Grid_V1
from algorithms.utils.q_learning_tools import Q_Table
from algorithms.mab_ucb import MAB_UCB_FG
import torch
from nets.small_net import CNN as CNN_SMALL
from nets.big_net import CNN as CNN_BIG
from nets.mask import CategoricalMasked
import multiprocessing
from enviroment.utils.final_reward import generate_reward


# net = CNN_SMALL(20, 1, 400, True)
# env = Full_Grid_V1(size=20)
# state = env.reset()
# net.forward(state)
# print(sum(dict((p.data_ptr(), p.numel()) for p in net.parameters()).values()))
# def square(x):
#     return x * x
# pool = multiprocessing.Pool()
# pool = multiprocessing.Pool(processes=2)
# inputs = [0,1]
# # outputs = pool.map(lambda x: generate_reward(n_sims = 10, size = 20, env_id = x), inputs)
# outputs = pool.map(square, inputs)
# print(outputs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

   



