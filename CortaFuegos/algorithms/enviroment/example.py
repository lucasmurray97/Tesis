from firegrid_v2 import FireGrid_V2
from parallel_firegrid import Parallel_Firegrid
env = Parallel_Firegrid(20)
env.reset()
env.show_state()
# for i in range(100):
#     env.step(env.random_action())
#     env.show_state()