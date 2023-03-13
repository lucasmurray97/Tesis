import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
file_baseline = open("demonstrations/Sub6x6_full_grid_1.pkl", 'rb')
stats_baseline = pickle.load(file_baseline)
sample = random.sample(range(0, 1000), 10)
for i in sample:
    mat = np.array(stats_baseline[i][0][4], dtype=np.int64)[0]*-1
    figure5 = plt.figure()
    plt.imshow(mat)
    plt.colorbar()
    plt.title(f"Agent's trajectory")
    plt.savefig(f"../../data/plots/baselines/sub6x6/baseline_6x6_{i}.png")
    plt.show()