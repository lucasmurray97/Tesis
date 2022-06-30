import matplotlib.pyplot as plt
from torch import nn as nn
import numpy as np

def plot_prog(env,episodes, policy,version, path,algorithm):
    state = env.reset()
    for i in range(100):
        mat = state[0].reshape(20,20).numpy()
        a = policy(state)
        f2 = plt.figure()
        plt.clf()
        plt.bar(np.arange(16), a.detach().numpy().squeeze())
        plt.xlabel("Actions") 
        plt.ylabel("Action Probability") 
        plt.title(f"Action probabilities in state {i} after training in trajectory of agent")
        plt.savefig(f"{path}/{version}/{algorithm}/{episodes}_ep/probabilities/post_train/probs_after_training_"+ str(i) +".png")
        plt.show()
        selected = a.multinomial(1).detach()
        state, done, _ = env.step(selected)
        figure5 = plt.figure()
        plt.imshow(mat)
        plt.colorbar()
        plt.title(f"State {i} in agent's trajectory")
        plt.savefig(f"{path}/{version}/{algorithm}/{episodes}_ep/trajectory/" + str(i) + ".png")
        plt.show()





