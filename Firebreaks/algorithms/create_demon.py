from algorithms.dpv.baseline import generate_demonstrations
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2

# Version 1
env = Full_Grid_V1(6)
generate_demonstrations(10000, 6, 1, 10, env, 1)

env = Full_Grid_V1(10)
generate_demonstrations(2000, 10, 1, 10, env, 1)

env = Full_Grid_V1(20)
generate_demonstrations(1000, 20, 1, 10, env, 1)

# # Version 2
# env = Full_Grid_V2(6)
# generate_demonstrations(10, 6, 1, 10, env, 2)

# env = Full_Grid_V2(10)
# generate_demonstrations(10, 10, 1, 10, env, 2)

# env = Full_Grid_V2(20)
# generate_demonstrations(10, 20, 1, 10, env, 2)