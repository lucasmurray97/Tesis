#!/bin/sh
## small 
# 0 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --prioritized


# 1000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 1000 --prioritized

# 10000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 1000 --demonstrate --n_dem 10000 --prioritized

## big
# 0 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --prioritized


# 1000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 1000 --prioritized

# 10000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --max_mem 20000  --target_update 200 --exploration_fraction 0.5 --pre_epochs 2000 --demonstrate --n_dem 10000 --prioritized