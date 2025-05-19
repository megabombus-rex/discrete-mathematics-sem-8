import networkx as nx
import numpy as np
from algorithms import *
from graph_generators import *
from drawing_tools import *

def main(maxcut:MaxCut, graph_size:int):
    graph = GraphGenerator().generate_simple_graph_temperature(graph_size, 10)
    
    (max_value, partition) = maxcut(graph)

    Painter.visualize_cut(graph,partition)
    print(f'Max count value is: {max_value}')

if __name__ == '__main__':
    main(BruteForceMaxCut(), 15)