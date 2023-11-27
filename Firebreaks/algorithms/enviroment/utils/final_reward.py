import sys
import numpy as np
from numpy import genfromtxt
import csv  
import pprint
import os
# We add the path to the simulator in order to execute it
sys.path.append("../../../../src/Cell2Fire/")
sys.path.append("../../src/cell2fire/Cell2Fire/")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import os
import shutil
import random
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def write_firewall_file(final_state, env_id = 0, instance = "homo_1"):
    """Function that writes the final state in the format delimited for firewalls in a file called HarvestedCells.csv"""
    i = 1
    firewalls = [1]
    for cell in final_state.flatten():
        if cell == -1:
            firewalls.append(i)
        i+=1
    header = ['Year Number','Cell Numbers']
    absolute_path = os.path.dirname(__file__)
    with open(f'{absolute_path}/instances/{instance}/firewall_grids/HarvestedCells_{env_id}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(firewalls)
    return
def generate_reward(n_sims, size, env_id = 0, instance = "homo_1"):
    """Function that generates the reward associated with the fire simulation"""
    absolute_path = os.path.dirname(__file__)
    data_directory = ""
    results_directory = ""
    base_size = 20 if size < 20 else size
    data_directory = f"{absolute_path}/instances/{instance}/data/Sub{base_size}x{base_size}_{env_id}/"
    results_directory = f"{absolute_path}/instances/{instance}/results/Sub{base_size}x{base_size}_{env_id}/"
    harvest_directory = f"{absolute_path}/instances/{instance}/firewall_grids/HarvestedCells_{env_id}.csv"
    try:
        shutil.rmtree(f'{results_directory}Grids/')
    except:
        pass
    # A command line input is simulated
    if instance == "homo_1":
        ros_cv = 1.0
        if size == 20:
            ignition_rad = 9
        elif size == 10:
            ignition_rad = 2
        else:
            ignition_rad = 0
    elif instance == "homo_2" or instance == "hetero_1":
        ros_cv = 0.0
        ignition_rad = 4
    else:
        ros_cv = 0.0
        ignition_rad = 9
    seed = random.randrange(0,10000)
    # n_weathers = 350
    n_weathers = len([i for i in os.listdir(data_directory+"Weathers/") if i.endswith('.csv')])-2
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', str(n_weathers), '--Fire-Period-Length', '1.0', '--ROS-CV', str(ros_cv), '--IgnitionRad', str(ignition_rad), '--HarvestedCells', harvest_directory, '--seed', str(seed)]
    # The main loop of the simulator is run for an instance of {base_size}x{base_size}
    blockPrint()
    main()
    enablePrint()
    # The grid from the final period is retrieved
    reward = 0
    base_directory = f"{absolute_path}/instances/{instance}/results/Sub{base_size}x{base_size}_{env_id}/Grids/Grids"
    for j in range(1, n_sims+1):
        dir = f"{base_directory}{str(j)}/"
        files = os.listdir(dir)
        my_data = genfromtxt(dir+files[-1], delimiter=',')
        # Burned cells are counted and turned into negative rewards
        for cell in my_data.flatten():
            if cell == 1:
                reward-= 1
    return (reward/n_sims)*(size/20)
