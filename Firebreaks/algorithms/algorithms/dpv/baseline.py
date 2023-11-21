from algorithms.dpv.calculate_dpv import calculate_dpv, blockPrint, enablePrint, main
from enviroment.utils.final_reward import write_firewall_file, generate_reward 
import random
import os
import csv
import sys
import shutil
import numpy as np
from numpy import genfromtxt
import torch
import cv2
import json
import pickle
from tqdm import tqdm
sys.path.append("../../")
absolute_path = os.path.dirname(__file__)
def baseline(size, n_sims, forbidden, instance):
    quant = int((size**2)*0.05)
    if quant%2 == 0:
        quant += 1
    erase_firebreaks(instance)
    seed = random.randint(0, 10000)
    firebreaks = [0]
    upscaled_firebreaks = [0]
    for _ in range(quant):
        firebreaks = calc_firebreaks(size, firebreaks, upscaled_firebreaks, seed, n_sims, forbidden, instance)
    for i in firebreaks[1:]:
        if i in forbidden:
            print(firebreaks[1:], forbidden)
            raise("Cell in forbidden!!")
    if len(list(set(firebreaks[1:]))) < len(firebreaks[1:]):
        print(list(set(firebreaks[1:])), firebreaks[1:])
        raise("Repeated firebreak!!")
    return firebreaks[1:]


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

def eval(size, n_sims, instance):
    return generate_reward(n_sims, size, instance = instance)

def simulate_episode(size, n_sims, n_sims_eval, forbidden, instance):
    solution = baseline(size, n_sims, forbidden, instance)
    reward = eval(size, n_sims_eval, instance)
    episode_data = []
    for i in range(len(solution)-1):
        episode_data.append([solution[i], 0, False])
    episode_data.append([solution[-1], reward, True])
    return episode_data

def calc_firebreaks(size, firebreaks, upscaled_firebreaks, seed, n_sims, forbidden, instance):
    dpvs = calculate_dpv(seed, n_sims, instance=instance)
    dpvs = dict(enumerate(list(dpvs.values()), 0))
    if size == 20:
        for j in forbidden:
            dpvs.pop(j, None)
        for k in firebreaks:
                dpvs.pop(k, None)
        firebreak = max(dpvs, key=dpvs.get)
        if dpvs[firebreak] == 0:
            firebreak = random.choice(list(dpvs.keys()))
        firebreaks.append(firebreak)
        write_firebreaks(map(lambda x:x+1, firebreaks), instance)
        return firebreaks
    else:
        values = list(dpvs.values())
        np_dpv = np.array(values, dtype='uint8')
        np_dpv = np_dpv.reshape((20,20))
        shrinked = cv2.resize(src = np_dpv, dsize=(size,size), interpolation = 1)
        new_dpvs = dict(enumerate(shrinked.flatten(), 0))
        for j in forbidden:
            new_dpvs.pop(j, None)
        for k in firebreaks[1:]:
            new_dpvs.pop(k, None)
        firebreak = max(new_dpvs, key=new_dpvs.get)
        if new_dpvs[firebreak] == 0:
            firebreak = random.choice(list(new_dpvs.keys()))
        firebreaks.append(firebreak)
        intermediate = np.zeros(size**2)
        intermediate[firebreak] = 1
        intermediate = intermediate.reshape((size, size))
        upscaled = cv2.resize(src = intermediate, dsize=(20,20), interpolation = 0)
        upscaled_flattened = dict(enumerate(upscaled.flatten(), 0))
        for i in upscaled_flattened.keys():
            if upscaled_flattened[i] == 1:
                upscaled_firebreaks.append(i)
        write_firebreaks(map(lambda x:x+1, upscaled_firebreaks), instance)
        return firebreaks

def generate_demonstrations(episodes, size, n_sims, n_sims_eval, env, version, instance = "homo_1"):
    data = {}
    history = {i: 0 for i in range(size**2)}
    forbidden = env.forbidden_cells
    for i in  tqdm(range(episodes)):
        episode = simulate_episode(size, n_sims, n_sims_eval, forbidden, instance)
        data[i] = {}
        step = 0
        state = env.reset()
        done = False
        j = 0
        while not done:
            step_data = [state.clone().tolist().copy(), episode[j][0], episode[j][1], episode[j][2]]
            history[episode[j][0]] += 1
            next_state, _, done = env.step(torch.Tensor([episode[j][0]]))
            step_data.append(next_state.clone().tolist().copy())
            if done:
                next_state = env.reset()
            data[i][step] = step_data
            state = next_state
            step += 1
            j += 1
    print(history)
    try:
        os.makedirs(f"algorithms/dpv/demonstrations/{instance}/")
    except OSError as error:  
        print(error)
    with open(f"algorithms/dpv/demonstrations/{instance}/Sub{size}x{size}_full_grid_{version}.pkl", "wb+") as write_file:
        pickle.dump(data, write_file)
    with open(f"algorithms/dpv/demonstrations/{instance}/history_Sub{size}x{size}_full_grid_{version}.pkl", "wb+") as write_file:
        pickle.dump(history, write_file)
    file = open(f"algorithms/dpv/demonstrations/{instance}/Sub{size}x{size}_full_grid_{version}.pkl", 'rb')    
    n_dems = pickle.load(file)
    print(f"Wrote {len(n_dems.keys())} demons in dpv/demonstrations/{instance}/Sub{size}x{size}_full_grid_{version}")










