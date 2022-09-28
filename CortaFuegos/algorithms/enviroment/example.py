from firegrid_v2 import FireGrid_V2
from firegrid_v6 import FireGrid_V6
from parallel_firegrid import Parallel_Firegrid
env = FireGrid_V6(20)
env.reset()
env.show_state()
for i in range(100):
    env.step(env.random_action())
    env.show_state()