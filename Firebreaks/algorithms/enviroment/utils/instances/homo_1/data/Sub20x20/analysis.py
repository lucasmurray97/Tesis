import sys
import os
import numpy as np
from numpy import genfromtxt
sys.path.append("/home/lucas/Tesis/Simulador/Cell2Fire/Cell2Fire/cell2fire")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import random
import shutil
import csv
import pickle
arr = os.listdir("Weathers_orig/")
n_rows = 9
for file in arr:
    if file[-3:] == "csv":
        with open(f"Weathers_orig/{file}", 'rt') as inp, open(f'Weathers/{file}', 'wt') as out:
            writer = csv.writer(out)
            n = 0
            for row in csv.reader(inp):
                if n < 4:
                    if n > 0:
                        row[4] = '60'
                        row[5] = '30'
                        row[6] = '315'
                    writer.writerow(row)
                    n += 1
absolute_path = os.path.dirname(__file__)
data_directory = f"{absolute_path}/"
results_directory = f"{absolute_path}/../../results/Sub20x20/"
harvest_directory = f"{absolute_path}/../../firewall_grids/HarvestedCells_0.csv"
try:
  shutil.rmtree(f'{results_directory}Grids/')
except:
  pass

file = open(f"../../../../../../algorithms/dpv/demonstrations/homo_1/history_Sub20x20_full_grid_1.pkl", 'rb')
history = pickle.load(file)
print(dict(sorted(history.items(), key = lambda x: x[1], reverse = True)[:21]))
ros_cv = 1.0
ignition_rad = 0
sims = 5000
seed = random.randrange(0,10000)
sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(sims), '--grids', '--weather', 'random', '--nweathers', '350', '--Fire-Period-Length', '1.0', '--ROS-CV', str(ros_cv), '--IgnitionRad', str(ignition_rad),'--seed', str(seed), '--HarvestedCells', harvest_directory, '--stats', '--allPlots', '--combine']
main()
base_directory = f"{results_directory}Grids/Grids"
reward = 0

for j in range(1, sims+1):
    dir = f"{base_directory}{str(j)}/"
    files = os.listdir(dir)
    my_data = genfromtxt(dir+files[-1], delimiter=',')
    # Burned cells are counted and turned into negative rewards
    for cell in my_data.flatten():
        if cell == 1:
            reward-= 1
print(-(reward/sims)/400)