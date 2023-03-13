from matplotlib.pylab import *
import networkx as nx
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
from algorithms.dpv import ReadDataPrometheus
from algorithms.dpv.DPV import downtream_protection_value
import shutil
sys.path.append("../../../../../Simulador/Cell2Fire/Cell2Fire/")
from cell2fire.Cell2FireC_class import Cell2FireC
from cell2fire.main import main

def blockPrint():
    sys.stdout = open(os.devnull, 'w')
def enablePrint():
    sys.stdout = sys.__stdout__

def calculate_dpv(seed, n_sims, instance = "homo_1"):
    absolute_path = os.path.dirname(__file__)
    """Function that generates the reward associated with the fire simulation"""
    data_directory = f"{absolute_path}/data_dpv/{instance}/Sub20x20/"
    results_directory = f"{absolute_path}/data_dpv/{instance}/Sub20x20/results/"
    harvest_directory = f"{absolute_path}/data_dpv/{instance}/Sub20x20/firebreaks/HarvestedCells.csv"
    try:
        shutil.rmtree(f'{results_directory}Grids/')
        shutil.rmtree(f'{results_directory}Messages/')
    except:
        pass
    # A command line input is simulated
    ignition_rad = 4 if instance == "homo_2" else 9
    sys.argv = ['main.py', '--input-instance-folder', data_directory, '--output-folder', results_directory, '--ignitions', '--sim-years', '1', '--nsims', str(n_sims), '--finalGrid', '--weather', 'random', '--nweathers', '10', '--Fire-Period-Length', '1.0', '--ROS-CV', '0.0', '--IgnitionRad', '9', '--grids', '--output-messages', '--HarvestedCells', harvest_directory, '--seed', str(seed) ]
    # The main loop of the simulator is run for an instance of 20x20
    blockPrint()
    main()
    enablePrint()
    # Se leen los datos del bosque
    Folder = f'{absolute_path}/data_dpv/{instance}/Sub20x20/'
    #FBPlookup = Folder + 'fbp_lookup_table.csv' # Diccionario
    FBPlookup = Folder + 'fbp_lookup_table.csv' # Diccionario
    ForestFile = Folder + 'Forest.asc'   # Combustible

    # Se lee lookup_table
    FBPDict, ColorsDict = ReadDataPrometheus.Dictionary(FBPlookup)
    # Se lee la capa de combustibles Forest.asc
    CellsGrid3, CellsGrid4, Rows, Cols, AdjCells, CoordCells, CellSide = ReadDataPrometheus.ForestGrid(ForestFile,FBPDict)

    NCells = Rows * Cols
    m = Rows
    n = Cols
    Colors = []

    # Aqui construimos un set de colores, para posteriormente graficar
    for i in range(NCells):
        if str(CellsGrid3[i]) not in ColorsDict.keys():
            Colors.append((1.0,1.0,1.0,1.0))
        if str(CellsGrid3[i]) in ColorsDict.keys():
            Colors.append(ColorsDict[str(CellsGrid3[i])])

    AvailSet = set()
    NonAvailSet = set()

    for i in range(NCells):
        if CellsGrid4[i] != 'NF':
            AvailSet.add(i+1)
        else:
            NonAvailSet.add(i+1)

    setDir = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW']
    aux = set([])

    Adjacents = {} # diccionario de celdas adyacentes disponibles (no condidera las NF)
    for k in range(NCells):
        aux = set([])
        for i in setDir:
            if AdjCells[k][i] != None:
                if AdjCells[k][i][0] in AvailSet :
                    aux = aux | set(AdjCells[k][i])
        Adjacents[k + 1] = aux & AvailSet

    FAG = nx.Graph() # Fuel Avail Graph
    FAG.add_nodes_from(list(AvailSet))

    coord_pos = dict()
    for i in AvailSet:
        coord_pos[i] = CoordCells[i-1]

    ColorsAvail = {}
    for i in AvailSet:
        FAG.add_edges_from([(i,j) for j in Adjacents[i]])
        ColorsAvail[i] = Colors[i-1]

    cmdoutput1 = os.listdir(Folder + 'results/Messages')
    if ".DS_Store" in cmdoutput1:
        idx = cmdoutput1.index('.DS_Store') # Busca el indice que hay que eliminar
        del cmdoutput1[idx]
    if "desktop.ini" in cmdoutput1:
        idx = cmdoutput1.index('desktop.ini') # Busca el indice que hay que eliminar
        del cmdoutput1[idx]
    pathmessages = Folder + 'results/Messages/'

    dpv_values = downtream_protection_value(FAG,pathmessages)
    return dpv_values
