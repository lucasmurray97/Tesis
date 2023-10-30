import os
import sys
sys.path.append("../../../../../Simulador/Cell2Fire/Cell2Fire/")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import shutil
import random
import csv
import numpy as np
import cv2
from numpy import genfromtxt
from tqdm import tqdm
import pickle
sys.path.append("../../")
from enviroment.utils.final_reward import write_firewall_file, generate_reward
seed = random.randint(0, 10000)
absolute_path = os.path.dirname(__file__)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

def erase_firebreaks(instance):
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/../../enviroment/utils/instances/{instance}/firewall_grids/HarvestedCells_0.csv"
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

def write_firebreaks(firebreaks, instance):
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/../../enviroment/utils/instances/{instance}/firewall_grids/HarvestedCells_0.csv"   
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

        writer.writerow(firebreaks)

def optimal_firebreaks(size, forbidden, instance):
    erase_firebreaks(instance)
    n_cells = size**2
    n_firebreaks = int((n_cells)*0.05)
    cell = 0
    firebreaks = []
    if size == 20:
        limit = 9  
    elif size == 10:
        limit = 2
    elif size == 6: 
        limit = 0
    cell = 0
    for i in range(size):
        for j in range(size):
           if (i == limit and j <= limit) or (j == limit and i < limit):
                firebreaks.append(cell)
           cell += 1
    firebreaks = np.array(firebreaks)
    if size < 20:
        shrinked = np.zeros(n_cells, dtype='uint8')
        shrinked[firebreaks] = 1
        upscaled = cv2.resize(src = shrinked.reshape(size,size), dsize=(20,20), interpolation = 0)
        upscaled_flattened = dict(enumerate(upscaled.flatten(), 0))
        upscaled_firebreaks = [0]
        for i in upscaled_flattened.keys():
            if upscaled_flattened[i] == 1:
                upscaled_firebreaks.append(i)
        write_firebreaks(map(lambda x:x+1, upscaled_firebreaks), instance)
        return firebreaks
    else:
        write_firebreaks(map(lambda x:x+1, np.insert(firebreaks, 0, 0)), instance)
        return firebreaks

def run_sim(size, env, instance,seed=seed, n_sims=10):
    """Function that generates the reward associated with the fire simulation"""
    forbidden = env.forbidden_cells
    firebreaks = optimal_firebreaks(size, forbidden, instance)
    reward = generate_reward(n_sims, size, instance = instance)
    return firebreaks, reward

def generate_solutions(episodes, size, env, instance, n_sims = 10, seed = seed):
    rewards = []
    for i in tqdm(range(episodes)):
        firebreaks, reward = run_sim(size, env,instance, n_sims)
        rewards.append(reward)
    with open(f"{absolute_path}/solutions/{instance}/Sub{size}x{size}_full_grid.pkl", "wb+") as write_file:
            pickle.dump(rewards, write_file)
    file = open(f"{absolute_path}/solutions/{instance}/Sub{size}x{size}_full_grid.pkl", 'rb')    
    n_r = pickle.load(file)
    print(len(n_r))