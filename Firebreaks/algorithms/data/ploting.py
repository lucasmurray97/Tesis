import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
size = 10
instance = "homo_2"
env_version = "v1"
net_version = "small"
demonstrations = True
if demonstrations:
    pre_epochs = 20000
    n_dem = 50000
else:
    pre_epochs = 0
    n_dem = 0
file_dqn = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/dqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=False_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_dqn = json.load(file_dqn)
file_2dqn = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/2dqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=False_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_2dqn = json.load(file_2dqn)
file_ddqn = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/ddqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=False_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_ddqn = json.load(file_ddqn)

file_dqn_p = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/dqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=True_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_dqn_p = json.load(file_dqn_p)
file_2dqn_p = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/2dqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=True_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_2dqn_p = json.load(file_2dqn_p)
file_ddqn_p = open(f"full_grid/{env_version}/{instance}/sub{size}x{size}/{net_version}/ddqn/stats_episodes=32000_alpha=5e-05_gamma=1.0_epsilon=1_target_update=200_prioritized=True_n_dem={n_dem}_exploration=0.5_pre_epochs={pre_epochs}_.json", 'r')
stats_ddqn_p = json.load(file_ddqn_p)

file_baseline = open(f"../algorithms/dpv/demonstrations/{instance}/Sub{size}x{size}_full_grid_1.pkl", 'rb')
stats_baseline = pickle.load(file_baseline)

file_random = open(f"../algorithms/random/solutions/{instance}/Sub{size}x{size}_full_grid.pkl", 'rb')
stats_random = pickle.load(file_random)

returns = []
for i in stats_baseline.keys():
    ep_ret = 0
    for j in stats_baseline[i].keys():
        ep_ret += stats_baseline[i][j][2]
        if stats_baseline[i][j][3]:
            returns.append(stats_baseline[i][j][2])
window = 5000
returns = [x for x in returns][:50000]
ret_dqn = np.cumsum(stats_dqn["Returns"], dtype=float)
x = np.linspace(0, 1, len(ret_dqn[window - 1:]))
ret_dqn[window:] = ret_dqn[window:] - ret_dqn[:-window]
ret_2dqn = np.cumsum(stats_2dqn["Returns"], dtype=float)
ret_2dqn[window:] = ret_2dqn[window:] - ret_2dqn[:-window]
ret_ddqn = np.cumsum(stats_ddqn["Returns"], dtype=float)
ret_ddqn[window:] = ret_ddqn[window:] - ret_ddqn[:-window]

ret_dqn_p = np.cumsum(stats_dqn_p["Returns"], dtype=float)
x = np.linspace(0, 1, len(ret_dqn_p[window - 1:]))
ret_dqn_p[window:] = ret_dqn_p[window:] - ret_dqn_p[:-window]
ret_2dqn_p = np.cumsum(stats_2dqn_p["Returns"], dtype=float)
ret_2dqn_p[window:] = ret_2dqn_p[window:] - ret_2dqn_p[:-window]
ret_ddqn_p = np.cumsum(stats_ddqn_p["Returns"], dtype=float)
ret_ddqn_p[window:] = ret_ddqn_p[window:] - ret_ddqn_p[:-window]

ret_baseline = np.cumsum(returns, dtype=float)
ret_baseline[window:] = ret_baseline[window:] - ret_baseline[:-window]
x2 = np.linspace(0, 1, len(ret_baseline[window - 1:]))

window2 = 1000
ret_random = np.cumsum(stats_random, dtype=float)
ret_random[window2:] = ret_random[window2:] - ret_random[:-window2]
x3 = np.linspace(0, 1, len(ret_random[window2 - 1:]))


plt.plot(np.linspace(0, 1, len(ret_dqn[window - 1:])), ret_dqn[window - 1:] / window, label = "dqn")
plt.plot(np.linspace(0, 1, len(ret_2dqn[window - 1:])), ret_2dqn[window - 1:] / window, label = "2dn")
plt.plot(np.linspace(0, 1, len(ret_ddqn[window - 1:])), ret_ddqn[window - 1:] / window, label = "ddqn")

plt.plot(np.linspace(0, 1, len(ret_dqn_p[window - 1:])), ret_dqn_p[window - 1:] / window, label = "dqn_p")
plt.plot(np.linspace(0, 1, len(ret_2dqn_p[window - 1:])), ret_2dqn_p[window - 1:] / window, label = "2dqn_p")
plt.plot(np.linspace(0, 1, len(ret_ddqn_p[window - 1:])), ret_ddqn_p[window - 1:] / window, label = "ddqn_p")

plt.plot(x2, ret_baseline[window - 1:] / window, label = "baseline")
plt.plot(x3, ret_random[window2 - 1:] / window2, label = "random")

plt.legend()
plt.xlabel("Episode") 
plt.ylabel("Average return") 
plt.title(f"Moving average of rewards, {n_dem} demonstrations")
plt.savefig(f"plots/{instance}/compare_{instance}_sub{size}x{size}_{env_version}_n_dem={n_dem}_returns_{net_version}.png")

# plt.show()

# mat = np.array(stats_baseline[i][j][0][0], dtype=int)*-1
# mat[2][0] = -1
# figure5 = plt.figure()
# plt.imshow(mat)
# plt.colorbar()
# plt.title(f"Agent's trajectory")
# plt.savefig(f"plots/compare_sub{size}x{size}_trajectory.png")
# plt.show()

# mat = np.zeros((20, 20), dtype=int)
# for i in range(10):
#     mat[10][i] = -1
#     mat[i][10] = -1
# mat[10][10] = -1

# figure5 = plt.figure()
# plt.imshow(mat)
# plt.colorbar()
# plt.title(f"Agent's trajectory")
# plt.savefig(f"plots/optimal_sub20x20_trajectory.png")
# plt.show()
