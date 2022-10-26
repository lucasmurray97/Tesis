import random
from tqdm import tqdm
from enviroment.firegrid_v6 import FireGrid_V6
from enviroment.utils.parallel_wrapper import Parallel_Wrapper
from utils.q_learning_tools import Q_Table_2

def q_learning(size, env, episodes, env_version, plot_episode = [], alpha = 1e-5, gamma = 0.99, n_envs = 8, epsilon = 0.15):
    q_table = Q_Table_2(size = size, alpha = alpha, gamma = gamma)
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
    return returns, q_table