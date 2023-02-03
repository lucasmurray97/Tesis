from algorithms.dpv.calculate_dpv import calculate_dpv, blockPrint, enablePrint, main
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
def baseline(size, n_sims, forbidden):
    quant = int((size**2)*0.05)
    if quant%2 == 0:
        quant += 1
    erase_firebreaks()
    seed = random.randint(0, 10000)
    firebreaks = [0]
    upscaled_firebreaks = [0]
    for _ in range(quant):
        firebreaks = calc_firebreaks(size, firebreaks, upscaled_firebreaks, seed, n_sims, forbidden)
    for i in firebreaks[1:]:
        if i in forbidden:
            print(firebreaks[1:], forbidden)
            raise("Cell in forbidden!!")
    if len(list(set(firebreaks[1:]))) < len(firebreaks[1:]):
        print(list(set(firebreaks[1:])), firebreaks[1:])
        raise("Repeated firebreak!!")
    return firebreaks[1:]


def erase_firebreaks():
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/data_dpv/Sub20x20/firebreaks/HarvestedCells.csv"   
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

def write_firebreaks(firebreaks):
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/data_dpv/Sub20x20/firebreaks/HarvestedCells.csv"   
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

        writer.writerow(firebreaks)

def eval(size, n_sims):
    data_directory = f"{absolute_path}/data_dpv/Sub20x20/"
    results_directory = f"{absolute_path}/data_dpv/Sub20x20/results/"
    harvest_directory = f"{absolute_path}/data_dpv/Sub20x20/firebreaks/HarvestedCells.csv"
    try:
        shutil.rmtree(f'{results_directory}Grids/')
        shutil.rmtree(f'{results_directory}Messages/')
    except:
        pass
    # A command line input is simulated
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '10', '--Fire-Period-Length', '1.0', '--ROS-CV', '0.0', '--IgnitionRad', '9', '--grids', '--output-messages', '--HarvestedCells', harvest_directory]
    # The main loop of the simulator is run for an instance of 20x20
    blockPrint()
    main()
    enablePrint()
    reward = 0
    base_directory = f"{results_directory}/Grids/Grids"
    for j in range(1, n_sims+1):
        directory = os.listdir(base_directory+str(j))
        numbers = []
        for i in directory:
            numbers.append(int(i.split("d")[1].split(".")[0]))
        maxi = "0"+str(max(numbers))
        my_data = genfromtxt(base_directory+str(j)+'/ForestGrid' + maxi +'.csv', delimiter=',')
        # Burned cells are counted and turned into negative rewards
        for cell in my_data.flatten():
            if cell == 1:
                reward-= 1
    return (reward/n_sims)

def simulate_episode(size, n_sims, n_sims_eval, forbidden):
    solution = baseline(size, n_sims, forbidden)
    reward = eval(size, n_sims_eval)
    episode_data = []
    for i in range(len(solution)-1):
        episode_data.append([solution[i], 0, False])
    episode_data.append([solution[-1], reward, True])
    return episode_data

def calc_firebreaks(size, firebreaks, upscaled_firebreaks, seed, n_sims, forbidden):
    dpvs = calculate_dpv(seed, n_sims)
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
        write_firebreaks(map(lambda x:x+1, firebreaks))
        return firebreaks
    else:
        values = list(dpvs.values())
        np_dpv = np.array(values, dtype='uint8')
        np_dpv = np_dpv.reshape((20,20))
        shrinked = cv2.resize(src = np_dpv, dsize=(size,size), interpolation = 1)
        new_dpvs = dict(enumerate(shrinked.flatten(), 0))
        for j in forbidden:
            new_dpvs.pop(j, None)
        for k in firebreaks:
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
        write_firebreaks(map(lambda x:x+1, upscaled_firebreaks))
        return firebreaks

def generate_demonstrations(episodes, size, n_sims, n_sims_eval, env, version):
    data = {}
    forbidden = env.forbidden_cells
    for i in  tqdm(range(episodes)):
        episode = simulate_episode(size, n_sims, n_sims_eval, forbidden)
        data[i] = {}
        step = 0
        state = env.reset()
        done = False
        j = 0
        while not done:
            step_data = [state.clone().tolist().copy(), episode[j][0], episode[j][1], episode[j][2]]
            next_state, _, done = env.step(torch.Tensor([episode[j][0]]))
            if done:
                next_state = env.reset()
            step_data.append(next_state.clone().tolist().copy())
            data[i][step] = step_data
            state = next_state
            step += 1
            j += 1
    with open(f"algorithms/dpv/demonstrations/Sub{size}x{size}_full_grid_{version}.pkl", "wb+") as write_file:
        pickle.dump(data, write_file)
    file = open(f"algorithms/dpv/demonstrations/Sub{size}x{size}_full_grid_{version}.pkl", 'rb')    
    n_dems = pickle.load(file)
    print(len(n_dems.keys()))

# env = Full_Grid_V2(6)
# generate_demonstrations(10, 6, 1, 10, env, 1)









