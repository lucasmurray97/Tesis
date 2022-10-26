from algorithms.q_learning import q_learning
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.firegrid_v7 import FireGrid_V7
from enviroment.firegrid_v8 import FireGrid_V8
from enviroment.utils.parallel_wrapper import Parallel_Wrapper 
from utils.plot_progress import plot_prog, plot_moving_av
import matplotlib.pyplot as plt
import sys
n_envs = 8
# We retrieve the arguments from standard input
_, size, env_version, episodes_, lr_, eps_, instance, window_ = sys.argv
size = int(size)
episodes = int(episodes_)
lr = float(lr_)
eps = float(eps_)
window = int(size)

# We create the enviroment
if env_version == "v6":
    env = Parallel_Wrapper(FireGrid_V6, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 50})
elif env_version == "v7":
    env = Parallel_Wrapper(FireGrid_V7, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 5, "n_sims_final" : 50})
elif env_version == "v8":
    env = Parallel_Wrapper(FireGrid_V8, n_envs = n_envs, parameters = {"size": size, "burn_value": 10, "n_sims" : 5, "n_sims_final" : 5})
else:
    raise("Non existent version of enviroment")

returns, q_table = q_learning(env, episodes, env_version, [], alpha = lr, epsilon = eps)

plot_moving_av(returns, episodes, env_version, "", "q_learning", window = window, drl = False, test = True, params = {"lr": lr, "eps": eps}, instance = instance)

state = env.envs[0].reset()
for i in range((size//2)**2):
    mat = env.envs[0]._space[0].reshape(size,size).numpy()
    selected = selected = q_table.max_action(state)
    state, done, _ = env.envs[0].step(selected)
    if i == (size//2)**2-1:
        figure5 = plt.figure()
        plt.imshow(mat)
        plt.colorbar()
        plt.title(f"Agent's trajectory after {episodes} episodes")
        plt.savefig(f"figures_tuning/{env_version}/{instance}/q_learning/trajectory_episodes={episodes}_lr={lr}_eps={eps}.png")
        plt.show()






