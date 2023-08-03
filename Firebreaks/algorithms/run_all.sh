#!/bin/sh
# v1
python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  --prioritized
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v1 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  --prioritized
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v1 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized

#  v2
python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  --prioritized
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v2 --net_version small --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 

python3.10 main.py --algorithm dqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5  --prioritized
python3.10 main.py --algorithm 2dqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized
python3.10 main.py --algorithm ddqn --size 6 --env full_grid --env_version v2 --net_version big --episodes 100 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 50000  --target_update 200 --exploration_fraction 0.5 --prioritized
