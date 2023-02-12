#!/bin/sh
# no dems
python3.10 main.py --algorithm dqn --size 20 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 5e-5 --gamma 1. --instance sub10x10 --test --max_mem 20000  --target_update 10 --epsilon 1. --epsilon_dec 0.01 --epsilon_min 0.005
