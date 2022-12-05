#!/bin/sh
# Full Grid
# Sub 6x6
#Q_Learning
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-2 --epsilon 0.1 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-3 --epsilon 0.1 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-4 --epsilon 0.1 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-5 --epsilon 0.1 --instance sub6x6

python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-2 --epsilon 0.15 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-3 --epsilon 0.15 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-4 --epsilon 0.15 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-5 --epsilon 0.15 --instance sub6x6

python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-2 --epsilon 0.2 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-3 --epsilon 0.2 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-4 --epsilon 0.2 --instance sub6x6
python3 algorithm_analysis.py --algorithm q_learning --size 6 --env full_grid --env_version v1 --episodes 1000 --window 100  --alpha 2e-5 --epsilon 0.2 --instance sub6x6