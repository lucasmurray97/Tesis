#!/bin/sh
# v1
python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5  --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5  --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v1 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000

# v2
python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5  --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v2 --net_version small --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000 
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --demonstrate --n_dem 100000 --pre_epochs 40000

python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5  --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm 2dqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000
python3.10 main.py --algorithm ddqn --size 20 --env full_grid --env_version v2 --net_version big --episodes 1000 --window 500  --alpha 5e-5 --gamma 1. --instance homo_1 --max_mem 200000  --target_update 200 --exploration_fraction 0.5 --prioritized --demonstrate --n_dem 100000 --pre_epochs 40000

