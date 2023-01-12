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
env = Full_Grid_V1(size = 20)
env_shape = (1, 20, 20)
memory = ReplayMemory(env_shape, max_mem=15, batch_size=10, demonstrate=True, n_dem=10, prioritized=True, env="FG", version=1, size=env_shape[1],n_envs=16, gamma = 1., landa = 1.)
print(memory.buffer.action_memory)
step = 0
state = env.reset()
action = env.random_action()
next_state, reward, done = env.step(action)
memory.buffer.store_transition(state, action.unsqueeze(0), reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))

action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
action = env.random_action()
memory.buffer.store_transition(state, action, reward, next_state, done, 0.99, 0.99, torch.Tensor([i for i in range(400)]).unsqueeze(0), torch.Tensor([1]).unsqueeze(0))
print(memory.buffer.action_memory)

print(memory.buffer.sample_memory()[0])


