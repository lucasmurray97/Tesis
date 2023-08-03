import sys
import os
import numpy as np
from numpy import genfromtxt
sys.path.append("/home/lucas/Tesis/Simulador/Cell2Fire/Cell2Fire/cell2fire")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import shutil
import csv
import random
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
                        row[4] = '80'
                        row[5] = '15'
                        # row[6] = '315'
                    writer.writerow(row)
                    n += 1
absolute_path = os.path.dirname(__file__)
data_directory = f"{absolute_path}/"
results_directory = f"{absolute_path}/../../results/Sub20x20/"
try:
  shutil.rmtree(f'{results_directory}Grids/')
except:
  pass
ignition_rad = 4
sims = 50
total_reward = 0
ros_cv = 0.0
for i in range(sims):
    n_sims = 50
    ros_cv = 0.0
    seed = random.randrange(0,10000)
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '350', '--Fire-Period-Length', '1.0', '--ROS-CV', str(ros_cv), '--IgnitionRad', str(ignition_rad), '--seed', str(seed)]
    main()
    base_directory = f"{results_directory}Grids/Grids"
    reward = 0
    for j in range(1, n_sims+1):
            my_data = genfromtxt(base_directory+str(j)+'/ForestGrid00.csv', delimiter=',')
            # Burned cells are counted and turned into negative rewards
            for cell in my_data.flatten():
                if cell == 1:
                    reward-= 1
    total_reward += -(reward/n_sims)
print((total_reward/sims)/400)