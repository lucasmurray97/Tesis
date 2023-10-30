import os
import sys
import argparse
import shutil

"""
Parsing of arguments
"""
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--size', type=int, required=True)
parser.add_argument('--data', type=str, required=True)
parser.add_argument('--n_cores', type=int, default=32)
args = parser.parse_args()
data_path = args.data
size = args.size
n_cores = args.n_cores
"""
Creating directories in environment
"""
path_env = f"../enviroment/utils/instances/{args.name}/"
try:  
    os.mkdir(path_env)
except OSError as error:  
    print(error)   
try:
    os.mkdir(path_env+"firewall_grids")
except OSError as error:  
    print(error)   
try:
    os.mkdir(path_env+f"results/")
except OSError as error:  
    print(error)  
try: 
    os.mkdir(path_env+f"results/Sub{size}x{size}")
except OSError as error:  
    print(error)   
for i in range(n_cores):
    try:
        os.mkdir(path_env+f"results/Sub{size}x{size}_{i}")
    except OSError as error:  
        print(error)   
destination_path = path_env + f"data/" 
try:
    shutil.copytree(data_path, destination_path+f"Sub{size}x{size}")
except OSError as error:  
    print(error)   
for i in range(n_cores):
    try:
        shutil.copytree(data_path, destination_path +f"Sub{size}x{size}_{i}")
    except OSError as error:  
        print(error)   
"""
Creating directories in data
"""
path_data = f"../data/full_grid/"
try:
    shutil.copytree(path_data+"v1/homo_1", path_data+"v1/" + args.name)
except OSError as error:  
    print(error)   
try:
    shutil.copytree(path_data+"v1/homo_1", path_data+"v2/" + args.name)
except OSError as error:  
    print(error)   
