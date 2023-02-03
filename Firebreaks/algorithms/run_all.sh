#!/bin/sh
python3.10 create_demon.py

# no dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --prioritized

# 1000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mam 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 1000 --prioritized

# 5000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 5000 --prioritized

# 10000 dems
python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000

python3.10 main.py --algorithm dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000

python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm ddqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized
python3.10 main.py --algorithm 2dqn --size 10 --env full_grid --env_version v1 --net_version big --episodes 300 --window 500  --alpha 5e-4 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005 --demonstrate --n_dem 10000 --prioritized