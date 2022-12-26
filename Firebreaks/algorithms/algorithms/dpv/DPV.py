import os
import networkx as nx

def downtream_protection_value(G, pathmessages):
    val_at_risk = dict([(i, 1) for i in list(G.nodes)])
    nx.set_node_attributes(G, val_at_risk, 'val_at_risk')

    cmdoutput1 = os.listdir(pathmessages)
    if ".DS_Store" in cmdoutput1:
        idx = cmdoutput1.index('.DS_Store')  # Busca el indice que hay que eliminar
        del cmdoutput1[idx]
    if "desktop.ini" in cmdoutput1:
        idx = cmdoutput1.index('desktop.ini')  # Busca el indice que hay que eliminar
        del cmdoutput1[idx]

    nodos = list(G.nodes)
    ANCells = len(nodos)
#    dpv_list = dict(zip(nodos, [[] for i in range(ANCells)]))

    for i in G.nodes:
        G.nodes[i]['dpv'] = 0

    for k in cmdoutput1:
        # Leemos la marca
        H = nx.read_edgelist(path=pathmessages + k,
                             delimiter=',',
                             create_using=nx.DiGraph(),
                             nodetype=int,
                             data=[('time', float), ('ros', float)])
        for i in H.nodes:
            H.nodes[i]['val_at_risk'] = G.nodes[i]['val_at_risk']
            H.nodes[i]['dpv'] = 0


        Arbol = dict()

        for i in H.nodes:
            Arbol[i] = nx.subgraph(H, {i} | nx.descendants(H, i))
            H.nodes[i]['dpv'] = sum([H.nodes[j]['val_at_risk'] for j in Arbol[i].nodes])
            G.nodes[i]['dpv'] = G.nodes[i]['dpv'] + H.nodes[i]['dpv']
#            dpv_list[i].append(H.nodes[i]['dpv'])

    dpv_values = dict(G.nodes(data='dpv'))
    return(dpv_values)
