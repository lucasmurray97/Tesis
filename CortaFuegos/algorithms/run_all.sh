#!/bin/sh
# Sub 2x2
# MAB UCB
#python3 analysis_mab_ucb.py 2 10000 sub2x2 100
# MAB GREEDY
#python3 analysis_mab_eps_greedy.py 2 10000 0.15 100 sub2x2

# Sub 4x4
# MAB GREEDY
python3 analysis_mab_eps_greedy.py 4 150000 0.15 50 sub4x4
# MAB UCB
python3 analysis_mab_ucb.py 4 100000 sub4x4 500
: '
# Sub 2x2
# q_learning2
#v6
python3 analysis_q_learning_2.py 2 v6 500 2e-2 0.1 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-3 0.1 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-4 0.1 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-5 0.1 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-6 0.1 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-7 0.1 sub2x2 100 > log.err > log.out

python3 analysis_q_learning_2.py 2 v6 500 2e-2 0.15 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-3 0.15 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-4 0.15 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-5 0.15 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-6 0.15 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-7 0.15 sub2x2 100 > log.err > log.out

python3 analysis_q_learning_2.py 2 v6 500 2e-2 0.2 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-3 0.2 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-4 0.2 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-5 0.2 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-6 0.2 sub2x2 100 > log.err > log.out
python3 analysis_q_learning_2.py 2 v6 500 2e-7 0.2 sub2x2 100 > log.err > log.out

# Sub 4x4
# q_learning2
#v6
python3 analysis_q_learning_2.py 4 v6 500 2e-2 0.1 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-3 0.1 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-4 0.1 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-5 0.1 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-6 0.1 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-7 0.1 sub4x4 100 > log.err > log.out

python3 analysis_q_learning_2.py 4 v6 500 2e-2 0.15 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-3 0.15 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-4 0.15 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-5 0.15 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-6 0.15 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-7 0.15 sub4x4 100 > log.err > log.out

python3 analysis_q_learning_2.py 4 v6 500 2e-2 0.2 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-3 0.2 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-4 0.2 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-5 0.2 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-6 0.2 sub4x4 100 > log.err > log.out
python3 analysis_q_learning_2.py 4 v6 500 2e-7 0.2 sub4x4 100 > log.err > log.out
'