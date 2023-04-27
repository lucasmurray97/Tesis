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
seed = random.randint(0, 10000)
absolute_path = os.path.dirname(__file__)
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

def erase_firebreaks(instance):
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/data_random/{instance}/Sub20x20/firebreaks/HarvestedCells.csv"   
    # We empty out the firebreaks file
    with open(path, 'w', encoding='UTF8') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(header)

def write_firebreaks(firebreaks, instance):
    header = ['Year Number','Cell Numbers']
    path = f"{absolute_path}/data_random/{instance}/Sub20x20/firebreaks/HarvestedCells.csv"   
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
        upscaled_flattened = dict(enumerate(upscaled.flatten(), 0))
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
    data_directory = f"{absolute_path}/data_random/{instance}/Sub20x20/"
    results_directory = f"{absolute_path}/data_random/{instance}/Sub20x20/results/"
    harvest_directory = f"{absolute_path}/data_random/{instance}/Sub20x20/firebreaks/HarvestedCells.csv"
    try:
        shutil.rmtree(f'{results_directory}Grids/')
        shutil.rmtree(f'{results_directory}Messages/')
    except:
        pass
    # A command line input is simulated
    ignition_rad = 4
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '10', '--Fire-Period-Length', '1.0', '--ROS-CV', '0.0', '--IgnitionRad', str(ignition_rad), '--grids', '--output-messages', '--HarvestedCells', harvest_directory, '--seed', str(seed) ]
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
    return firebreaks, (reward/n_sims)*(size/20)*10

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