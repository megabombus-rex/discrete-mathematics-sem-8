import networkx as nx
import numpy as np
from graph_creation.rud_reader import RudyGraphReader
from graph_creation.graph_generators import GraphGenerator
from algorithms import *
from graph_creation import *
from drawing_tools import *
from simulation.experiments import *

def main(maxcut:MaxCutSolver, graph_size:int):
    #graph = GraphGenerator.generate_simple_graph_temperature(graph_size, 10)
    graph1 = RudyGraphReader.read_rud("data\\set1\\g1.rud")
    #graph2 = RudyGraphReader.read_rud("data\\set1\\g2.rud")
    #graph3 = RudyGraphReader.read_rud("data\\set1\\g3.rud")
    exp = Experiment()
    
    exp(graph1, maxcut, 2)
    #Painter.visualize_graph(graph)
    
    #(max_value, partition) = maxcut(graph)

#    Painter.visualize_cut(graph,partition)
    #print(f'Max count value is: {max_value}')

if __name__ == '__main__':
    main(GuirobiMaxCutSolver(), 10)