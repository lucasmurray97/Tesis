import sys
import os
import numpy as np
from numpy import genfromtxt
sys.path.append("/home/lucas/Tesis/Simulador/Cell2Fire/Cell2Fire/cell2fire")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main
import shutil
import csv
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
try:
  shutil.rmtree(f'{results_directory}Grids/')
except:
  pass
n_sims = 10
ignition_rad = 9
sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '10', '--Fire-Period-Length', '1.0', '--ROS-CV', '0.0', '--IgnitionRad', str(ignition_rad),'--stats', '--allPlots', '--grids', '--output-messages', '--combine']
main()
base_directory = f"{results_directory}Grids/Grids"
reward = 0
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
print(-(reward/n_sims)/400)