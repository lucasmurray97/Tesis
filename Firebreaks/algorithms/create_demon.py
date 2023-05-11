from algorithms.dpv.baseline import generate_demonstrations
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2

instance = "homo_1"
# Version 1
# env = Full_Grid_V1(6, instance=instance)
# generate_demonstrations(1000, 6, 1, 10, env, 1, instance)

# env = Full_Grid_V1(10, instance=instance)
# generate_demonstrations(10000, 10, 1, 10, env, 1, instance)

env = Full_Grid_V1(20, instance=instance)
generate_demonstrations(5000, 20, 1, 10, env, 1, instance)

# # Version 2
# env = Full_Grid_V2(6, instance=instance)
# generate_demonstrations(1000, 6, 1, 10, env, 2, instance)

# env = Full_Grid_V2(10, instance=instance)
# generate_demonstrations(10000, 10, 1, 10, env, 2, instance)

env = Full_Grid_V2(20, instance=instance)
generate_demonstrations(5000, 20, 1, 10, env, 2, instance)