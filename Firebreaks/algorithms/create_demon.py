from algorithms.dpv.baseline import generate_demonstrations
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2

# instance = "hetero_1"
# # Version 1

# env = Full_Grid_V1(20, instance=instance)
# generate_demonstrations(5000, 20, 1, 10, env, 1, instance)

instance = "hetero_2"
# Version 1
env = Full_Grid_V1(40, instance=instance)
generate_demonstrations(10, 40, 1, 10, env, 1, instance)

# env = Full_Grid_V1(20, instance=instance)
# generate_demonstrations(5000, 20, 1, 10, env, 1, instance)



