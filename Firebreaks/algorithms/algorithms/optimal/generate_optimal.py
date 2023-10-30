from optimal_algorithm import generate_solutions
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2
from tqdm import tqdm
env = Full_Grid_V1(10)
generate_solutions(3000, 10, env, instance = "homo_1", n_sims=10)
# env = Full_Grid_V1(10)
# generate_solutions(3000, 10, env, instance = "homo_1", n_sims=10)
# env = Full_Grid_V1(6)
# generate_solutions(100, 6, env, instance = "homo_1", n_sims=10)



