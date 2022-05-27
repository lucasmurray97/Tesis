import sys
from cell2fire.Cell2FireC_class import Cell2FireC
sys.path.insert(1, "../../../Simulador/Cell2Fire/cell2fire")
import numpy as np
from numpy import genfromtxt
import csv  
from main import main
import os
def write_firewall_file(final_state):
    """Function that writes the final state in the format delimited for firewalls in a file called HarvestedCells.csv"""
    i = 1
    firewalls = [1]
    for cell in final_state.flatten():
        if cell == -1:
            firewalls.append(i)
        i+=1
    header = ['Year Number','Cell Numbers']
    with open('utils/firewall_grids/HarvestedCells.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(firewalls)
    return
def generate_reward():
    """Function that generates the reward associated with the fire simulation"""
    # A command line input is simulated
    sys.argv = ['main.py', '--input-instance-folder', '../../Simulador/Cell2Fire/data/Sub20x20/', '--output-folder', 'utils/results/Sub20x20', '--ignitions', '--sim-years', '1', '--nsims', '1', '--finalGrid', '--weather', 'random', '--nweathers', '1', '--Fire-Period-Length', '1.0', '--output-messages', '--ROS-CV', '0.0', '--IgnitionRad', '5', '--grids', '--combine', '--HarvestedCells', 'utils/firewall_grids/HarvestedCells.csv']
    # The main loop of the simulator is run for an instance of 20x20
    main()
    # The grid from the final period is retrieved
    directory = os.listdir("utils/results/Sub20x20/Grids/Grids1")
    numbers = []
    for i in directory:
        numbers.append(int(i.split("d")[1].split(".")[0]))
    maxi = "0"+str(max(numbers))
    my_data = genfromtxt('utils/results/Sub20x20/Grids/Grids1/ForestGrid' + maxi +'.csv', delimiter=',')
    # Burned cells are counted and turned into negative rewards
    reward = 0
    for cell in my_data.flatten():
        if cell == 1:
            reward-= 1
    return reward