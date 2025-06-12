import networkx as nx
import numpy as np
from graph_creation.rud_reader import RudyGraphReader
from graph_creation.graph_generators import GraphGenerator
from algorithms import *
from graph_creation import *
from drawing_tools import *
from simulation.experiments import *
import os
#from analysis.data_analysis import compare_algorithms_from_multiple_csvs, compare_algorithms
from analysis.max_cut_analyzer import MaxCutAnalyzer

def generate_graphs():
    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(20, 10)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(20, 5)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(20, 2)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(15, 10)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(15, 5)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(15, 2)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(10, 10)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(10, 5)

    for i in range(20):
        GraphGenerator.generate_simple_graph_temperature(10, 2)
        
def test(maxcut:MaxCutSolver):    
    #generate_graphs()    
    exp = Experiment()    
    
    files = [f for f in os.listdir('data/set6') if os.path.isfile(os.path.join('data/set6', f))]
    for f in range(len(files)):
        files[f] = 'data/set6/' + files[f]
        print(files[f])
    
    for file in files:
        graph = RudyGraphReader.read_rud(file)
        exp(graph, file, maxcut, 10)    
    
    if isinstance(maxcut, GoemansWilliamsonMaxCutSolver) or isinstance(maxcut, QAOACirqMaxCutSolver):
        for i in range(1, 10):
            graph = RudyGraphReader.read_rud(f"data\\set4\\bqp-50-{i}.txt")
            exp(graph, maxcut, 10)
            
        for i in range(1, 10):
            graph = RudyGraphReader.read_rud(f"data\\set2\\sg3dl05{i}000.mc")
            exp(graph, maxcut, 10)
            
        # only 800 nodes instances
        for i in range(1, 21):
            graph = RudyGraphReader.read_rud(f"data\\set1\\g{i}.rud")
            exp(graph, maxcut, 10)
    
def analyze(results_path):
    analyzer = MaxCutAnalyzer(results_path)
    results = analyzer.run_complete_analysis()

def main(maxcut:MaxCutSolver):
    test(maxcut)  
    analyze('results/2025-06-04_results.csv')
    #Painter.visualize_graph(graph)
    
    #(max_value, partition) = maxcut(graph)

    #Painter.visualize_cut(graph,partition)
    #print(f'Max count value is: {max_value}')


if __name__ == '__main__':
    #main(GuirobiMaxCutSolver())
    #main(BruteForceMaxCutSolver())
    #main(GoemansWilliamsonMaxCutSolver())
    main(QAOACirqMaxCutSolver())