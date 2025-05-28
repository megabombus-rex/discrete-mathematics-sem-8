import networkx as nx
import numpy as np
from graph_creation.rud_reader import RudyGraphReader
from graph_creation.graph_generators import GraphGenerator
from algorithms import *
from graph_creation import *
from drawing_tools import *
from simulation.experiments import *

def main(maxcut:MaxCutSolver):
    exp = Experiment()
    #graph = GraphGenerator.generate_simple_graph_temperature(graph_size, 10)
    #graph1 = RudyGraphReader.read_rud("data\\set1\\g1.rud")
    #graph2 = RudyGraphReader.read_rud("data\\set1\\g2.rud")
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl051000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl052000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl053000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl054000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl055000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl056000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl057000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl058000.mc")
    exp(graph, maxcut, 10)
    graph = RudyGraphReader.read_rud("data\\set2\\sg3dl059000.mc")
    exp(graph, maxcut, 10)
    
    
    #Painter.visualize_graph(graph)
    
    #(max_value, partition) = maxcut(graph)

#    Painter.visualize_cut(graph,partition)
    #print(f'Max count value is: {max_value}')

if __name__ == '__main__':
    #main(GuirobiMaxCutSolver(), 10)
    main(BruteForceMaxCutSolver())