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
from nets.big_net_v1 import CNN_BIG_V1
from nets.big_net_v2 import CNN_BIG_V2
from nets.mask import CategoricalMasked
import multiprocessing
from enviroment.utils.final_reward import generate_reward
from algorithms.utils.replay_buffer import ReplayMemory

# net = CNN_SMALL_V2(6, 2, 36, True)
# net.to('cuda')
env = Parallel_Wrapper(Full_Grid_V1, n_envs = 16, parameters = {"size": 20})
memory = ReplayMemory(env.env_shape, max_mem=1000, batch_size=64, temporal=False)
memory.load_demonstrations()
# print(memory.buffer.sample_memory())
# for i in range(1):
#     state = env.reset()
#     done = False
#     step = 0
#     while not done:
#         a = env.random_action()
#         s, r, done = env.step(a)
#         memory.buffer.store_transition(s.to('cpu'), a.to('cpu'), r.to('cpu'), s.to('cpu'), step, done)
#         step += 1
# s, a, r, s_ = memory.buffer.sample_memory()
# print(s.shape, a.shape, r.shape, s_.shape)
# print(a)
# print(r)
# print(sum(dict((p.data_ptr(), p.numel()) for p in net.parameters()).values()))
# def square(x):
#     return x * x
# pool = multiprocessing.Pool()
# pool = multiprocessing.Pool(processes=2)
# inputs = [0,1]
# # outputs = pool.map(lambda x: generate_reward(n_sims = 10, size = 20, env_id = x), inputs)
# outputs = pool.map(square, inputs)
# print(outputs)

# env = Full_Grid_V2(size=20)
# state = env.reset()
# done = False
# while not done:
#     state, r, done = env.step(env.random_action())
#     print(r)


