import pickle
import numpy as np
import matplotlib.pyplot as plt
import random
size = 10
instance = "homo_2"
quant = int((size**2)*0.05)
if quant%2 == 0:
    quant += 1
file_baseline = open(f"demonstrations/{instance}/Sub{size}x{size}_full_grid_1.pkl", 'rb')
stats_baseline = pickle.load(file_baseline)
sample = random.sample(range(0, 1000), 10)
for i in sample:
    mat = np.array(stats_baseline[i][quant-1][4], dtype=np.int64)[0]*-1
    figure5 = plt.figure()
    plt.imshow(mat)
    plt.colorbar()
    plt.title(f"Agent's trajectory")
    plt.savefig(f"../../data/plots/baselines/{instance}/sub{size}x{size}/baseline_{size}x{size}_{i}.png")
    # plt.show()