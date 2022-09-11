from cnn_policy_value_v2 import CNN
from enviroment.firegrid_v4 import FireGrid_V4
import torch
env = FireGrid_V4(20, burn_value=10, n_sims=50)
start_state = env.reset()
net = CNN()
net.load_state_dict(torch.load("./weights/batched_reinforce_baseline.pth"))
print(net.forward(start_state))
