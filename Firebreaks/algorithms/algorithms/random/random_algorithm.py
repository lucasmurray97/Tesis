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
from tqdm import tqdm
sys.path.append("../../")
from enviroment.full_grid_v1 import Full_Grid_V1
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
def random_firebreaks(size, forbidden, instance):
    erase_firebreaks(instance)
    n_cells = size**2
    n_firebreaks = int((n_cells)*0.05)
    available_cells = [i for i in range(n_cells)]
    for i in forbidden:
        available_cells.remove(i)
    firebreaks = np.random.choice(available_cells, size=n_firebreaks, replace=False)
    if size < 20:
        shrinked = np.zeros(n_cells, dtype='uint8')
        shrinked[firebreaks] = 1
        upscaled = cv2.resize(src = shrinked.reshape(size,size), dsize=(20,20), interpolation = 0)
        upscaled_flattened = dict(enumerate(upscaled.flatten(), 1))
        upscaled_firebreaks = []
        for i in upscaled_flattened.keys():
            if upscaled_flattened[i] == 1:
                upscaled_firebreaks.append(i)
        write_firebreaks([0] + upscaled_firebreaks, instance)
        return firebreaks
    else:
        write_firebreaks([0] + firebreaks, instance)
        return firebreaks

def run_sim(size, env, instance,seed=seed, n_sims=10):
    """Function that generates the reward associated with the fire simulation"""
    forbidden = env.forbidden_cells
    firebreaks = random_firebreaks(size, forbidden, instance)
    reward = generate_reward(n_sims, size, instance = instance)
    return firebreaks, reward

def generate_solutions(episodes, size, env, instance, n_sims = 10, seed = seed):
    rewards = []
    for i in tqdm(range(episodes)):
        firebreaks, reward = run_sim(size, env,instance, n_sims)
        rewards.append(reward)
    try:
        os.makedirs(f"{absolute_path}/solutions/{instance}/")
    except OSError as error:  
        print(error)
    with open(f"{absolute_path}/solutions/{instance}/Sub{size}x{size}_full_grid.pkl", "wb+") as write_file:
            pickle.dump(rewards, write_file)
    file = open(f"{absolute_path}/solutions/{instance}/Sub{size}x{size}_full_grid.pkl", 'rb')    
    n_r = pickle.load(file)
    print(len(n_r))


def generate_complete_random(env, size, instance, n_sims = 10):
    forbidden = env.forbidden_cells
    erase_firebreaks(instance)
    n_cells = size**2
    n_firebreaks = int((n_cells)*0.05)
    available_cells = [i for i in range(n_cells)]
    for i in forbidden:
        available_cells.remove(i)
    firebreaks = np.random.choice(available_cells, size=n_firebreaks, replace=False)
    if size < 20:
        shrinked = np.zeros(n_cells, dtype='uint8')
        shrinked[firebreaks] = 1
        upscaled = cv2.resize(src = shrinked.reshape(size,size), dsize=(20,20), interpolation = 0)
        upscaled_flattened = dict(enumerate(upscaled.flatten(), 1))
        upscaled_firebreaks = []
        for i in upscaled_flattened.keys():
            if upscaled_flattened[i] == 1:
                upscaled_firebreaks.append(i)
        write_firebreaks([0] + upscaled_firebreaks, instance)
    else:
        write_firebreaks([0] + firebreaks, instance)
    state = np.zeros(n_cells, dtype='uint8')
    state[firebreaks] = 1
    state = state.reshape(size,size)
    reward = generate_reward(n_sims, size, instance = instance)
    return state, reward

def generate_solutions_complete(observations, env, size, instance, n_sims = 10):
    data = []
    for i in tqdm(range(observations)):
        state, evaluation = generate_complete_random(env, size, instance, n_sims)
        data.append([state, evaluation])
    try:
        os.makedirs(f"{absolute_path}/complete_random/{instance}/")
    except OSError as error:  
        print(error)
    with open(f"{absolute_path}/complete_random/{instance}/Sub{size}x{size}_full_grid.pkl", "wb+") as write_file:
            pickle.dump(data, write_file)
    file = open(f"{absolute_path}/complete_random/{instance}/Sub{size}x{size}_full_grid.pkl", 'rb')    
    n_r = pickle.load(file)
    print(len(n_r))
