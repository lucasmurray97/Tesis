from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
from enviroment.full_grid_v1 import Full_Grid_V1
from algorithms.utils.q_learning_tools import Q_Table
from algorithms.mab_ucb import MAB_UCB_FG
import torch
from nets.small_net import CNN
from nets.mask import CategoricalMasked
env = Full_Grid_V1(size = 20)
net = CNN(env.size, 1, 400, True)
state = env.reset()
done = False
policy, value = net.forward(state)
print(policy)
mask = torch.zeros(400, dtype=torch.bool)
mask[0] = True
mask[1] = True
head_masked = CategoricalMasked(logits=policy, mask = mask)
print(head_masked.probs)





