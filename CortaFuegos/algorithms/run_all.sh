#!/bin/sh
# Full Grid
# Sub 6x6
# MAB GREEDY
#python3 algorithm_analysis.py --algorithm mab_greedy --size 6 --env full_grid --env_version v1 --episodes 100 --window 10 --instance sub6x6
# MAB UCB
#python3 algorithm_analysis.py --algorithm mab_ucb --size 6 --env full_grid --env_version v1 --episodes 100 --window 100 --epsilon 0.1 --instance sub6x6
#Q_Learning
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 50 --alpha 2e-4 --epsilon 0.1 --instance sub6x6