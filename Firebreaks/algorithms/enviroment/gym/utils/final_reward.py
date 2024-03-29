import sys
import numpy as np
from numpy import genfromtxt
import csv  
# We add the path to the simulator in order to execute it
sys.path.append("/home/lucas/Tesis/Simulador/Cell2Fire/Cell2Fire/cell2fire")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import os
import shutil
def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__
def write_firewall_file(final_state, env_id = 0):
    """Function that writes the final state in the format delimited for firewalls in a file called HarvestedCells.csv"""
    i = 1
    firewalls = [1]
    for cell in final_state.flatten():
        if cell == -1:
            firewalls.append(i)
        i+=1
    header = ['Year Number','Cell Numbers']
    with open(f'/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/firewall_grids/HarvestedCells_{env_id}.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(firewalls)
    return
def generate_reward(n_sims, size, env_id = 0):
    """Function that generates the reward associated with the fire simulation"""
    data_directory = ""
    results_directory = ""
    data_directory = f"/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/data/Sub20x20_{env_id}/"
    results_directory = f"/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/results/Sub20x20_{env_id}/"
    harvest_directory = f"/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/firewall_grids/HarvestedCells_{env_id}.csv"
    try:
        shutil.rmtree(f'{results_directory}Grids/')
    except:
        pass
    # A command line input is simulated
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '10', '--Fire-Period-Length', '1.0', '--ROS-CV', '0.0', '--IgnitionRad', '9', '--grids', '--HarvestedCells', harvest_directory]
    # The main loop of the simulator is run for an instance of 20x20
    blockPrint()
    main()
    enablePrint()
    # The grid from the final period is retrieved
    reward = 0
    base_directory = f"/home/lucas/Tesis/CortaFuegos/algorithms/enviroment/utils/results/Sub20x20_{env_id}/Grids/Grids"
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
    return (reward/n_sims)*(size/20)