from cnn_policy_value_v2 import CNN
from enviroment.firegrid_v5 import FireGrid_V5
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v4 import FireGrid_V4
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
import torch
env = FireGrid_V8(20)
print(env._action_map)
# for j in range(2):
#     start_state = env.reset()
#     for i in range(100):
#         action = env.random_action()
#         print(action)
#         next_state, reward, done = env.step(action)
#         print(done)
#         print(reward)
#         print(env._space[0])
#         print(env.last_complete)
#         if j == 1:
#             quit()
# env.show_state()

