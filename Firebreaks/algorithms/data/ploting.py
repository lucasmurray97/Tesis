import matplotlib.pyplot as plt
import json
import numpy as np
import pickle
file_dqn = open(f"full_grid/sub6x6/v1/small/dqn/stats_episodes=4800_alpha=0.005_gamma=0.999_landa=1_beta=0.1_.json", 'r')
stats_dqn = json.load(file_dqn)
file_2dqn = open(f"full_grid/sub6x6/v1/small/2dqn/stats_episodes=4800_alpha=0.005_gamma=0.999_landa=1_beta=0.1_ .json", 'r')
stats_2dqn = json.load(file_2dqn)
file_ddqn = open(f"full_grid/sub6x6/v1/small/ddqn/stats_episodes=4800_alpha=0.005_gamma=0.999_landa=1_beta=0.1_ .json", 'r')
stats_ddqn = json.load(file_ddqn)

file_baseline = open("full_grid/sub6x6/Sub6x6_full_grid_1.pkl", 'rb')
stats_baseline = pickle.load(file_baseline)
returns = []
for i in stats_baseline.keys():
    for j in stats_baseline[i].keys():
        returns.append(stats_baseline[i][j][2])
window = 500
returns = [x * 10 for x in returns]
ret_dqn = np.cumsum(stats_dqn["Returns"], dtype=float)
ret_dqn[window:] = ret_dqn[window:] - ret_dqn[:-window]
x = np.linspace(0, 1, len(ret_dqn[window - 1:]))
ret_2dqn = np.cumsum(stats_2dqn["Returns"], dtype=float)
ret_2dqn[window:] = ret_2dqn[window:] - ret_2dqn[:-window]
ret_ddqn = np.cumsum(stats_ddqn["Returns"], dtype=float)
ret_ddqn[window:] = ret_ddqn[window:] - ret_ddqn[:-window]
ret_baseline = np.cumsum(returns*10, dtype=float)
ret_baseline[window:] = ret_baseline[window:] - ret_baseline[:-window]
x2 = np.linspace(0, 1, len(ret_baseline[window - 1:]))
plt.plot(x, ret_dqn[window - 1:] / window, label = "dqn")
plt.plot(x, ret_2dqn[window - 1:] / window, label = "2dn")
plt.plot(x, ret_ddqn[window - 1:] / window, label = "ddqn")
plt.plot(x2, ret_baseline[window - 1:] / window, label = "baseline")
plt.legend()
plt.xlabel("Episode") 
plt.ylabel("Average return") 
plt.title(f"Moving average of rewards")
plt.savefig(f"plots/compare_sub6x6.png")
plt.show()