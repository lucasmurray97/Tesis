from random_algorithm import generate_solutions, generate_solutions_complete
from enviroment.full_grid_v1 import Full_Grid_V1
from enviroment.full_grid_v2 import Full_Grid_V2
from tqdm import tqdm
import sys
sys.path.append("../../")
env = Full_Grid_V1(20)    
generate_solutions(10000, 20, env, 'hetero_1', n_sims=50)

# env = Full_Grid_V1(10)    
# generate_solutions_complete(30, env, 10, 'hetero_1', n_sims=50)
