#!/bin/sh
# Full Grid
# Sub 6x6
#PPO
# V1

python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-3 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-4 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-5 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True

python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-3 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-4 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v1 --net_version small --episodes 300 --window 500  --alpha 2e-5 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True


# V2

python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-3 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-4 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-5 --gamma 1. --landa 1. --beta 0.1 --instance sub6x6 --test True

python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-3 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-4 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True
python3 algorithm_analysis.py --algorithm ppo --size 6 --env full_grid --env_version v2 --net_version small --episodes 300 --window 500  --alpha 2e-5 --gamma 1. --landa 1. --beta 0.2 --instance sub6x6 --test True

