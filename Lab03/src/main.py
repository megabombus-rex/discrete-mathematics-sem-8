import networkx as nx
import numpy as np
from algorithms import *

def main(maxcut:MaxCut, graph_size:int):
    G = nx.Graph()
    G.add_nodes_from(np.arange(0,graph_size,1))
    # generate nodes
    G.add_edge()
    maxcut(G)

if __name__ == '__main__':
    main(BruteForceMaxCut())