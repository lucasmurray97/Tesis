import sys

from cell2fire.Cell2FireC_class import Cell2FireC
sys.path.insert(1, "../../../Simulador/Cell2Fire/cell2fire")
import main as main

sys.argv = ['main.py', '--input-instance-folder', '../../../Simulador/Cell2Fire/data/Sub20x20/', '--output-folder', 'results/Sub20x20', '--ignitions', '--sim-years', '1', '--nsims', '1', '--finalGrid', '--weather', 'random', '--nweathers', '1', '--Fire-Period-Length', '1.0', '--output-messages', '--ROS-CV', '0.0', '--seed', '123', '--IgnitionRad', '5', '--grids', '--combine']
main.main()