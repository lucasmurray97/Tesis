import random
from tqdm import tqdm
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
from algorithms.utils.q_learning_tools import Q_Table_2, Q_Table
from algorithms.utils.plot_progress import plot_moving_av
import matplotlib.pyplot as plt
def q_learning(size, env, episodes, env_version, plot_episode = [], alpha = 1e-5, gamma = 0.99, n_envs = 8, epsilon = 0.15, instance = "sub2x2", window = 10):
    if env.envs[0].name == "moving_grid":
        q_table = Q_Table_2(size = size, alpha = alpha, gamma = gamma, epsilon = epsilon)
    else:
        q_table = Q_Table(size = size, alpha = alpha, gamma = gamma, epsilon = epsilon)
    returns = []
    for episode in tqdm(range(1, episodes + 1)):
        done = False
        state = env.reset()
        ep_return = 0
        step = 0
        while not done:
            n_states = q_table.find_state(state, step)
            action = q_table.pick_greedy_action(n_states, step)
            next_state, reward, done = env.step(action)
            q_table.update_table(n_states, step, action, next_state, reward)
            step += 1
            ep_return += reward
        returns.extend(ep_return.squeeze().tolist())
    plot_moving_av(env.envs[0], returns, episodes, env_version, "", "q_learning_2", window = window, drl = False, test = True, params = {"lr": alpha, "eps": epsilon}, instance = instance)
    state = env.envs[0].reset()
    step = 0
    for i in range(env.envs[0].get_episode_len()):
        mat = env.envs[0]._space[0].reshape(size,size).to('cpu').numpy()
        selected = q_table.max_action(state, step)
        state, done, _ = env.envs[0].step(selected)
        step += 1
        if i == env.envs[0].get_episode_len()-1:
            figure5 = plt.figure()
            plt.imshow(mat)
            plt.colorbar()
            plt.title(f"Agent's trajectory after {episodes} episodes")
            plt.savefig(f"figures_tuning/{env.envs[0].get_name()}/{env_version}/{instance}/q_learning_2/trajectory_episodes={episodes}_lr={alpha}_eps={epsilon}.png")
            plt.show()
    return returns, q_table