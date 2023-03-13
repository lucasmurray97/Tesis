from random_algorithm import generate_solutions
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2
from tqdm import tqdm
# env = Full_Grid_V1(10)
# generate_solutions(10000, 10, env, n_sims=10)
env = Full_Grid_V1(20)
generate_solutions(10000, size = 20, env = env, n_sims=10)

