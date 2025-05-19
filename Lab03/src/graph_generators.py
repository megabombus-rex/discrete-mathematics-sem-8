import networkx as nx
import numpy as np
import math
import random

class GraphGenerator():
    def generate_simple_graph_temperature(size, temperature):
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(0,size,1))
        
        line = [(i - 1, i) for i in size - 1]
        
        max_edges = int(math.floor(temperature * size))
        
        edges = [(random.randint(0, size), random.randint(0, size)) for i in max_edges]
        line.append(edges)
        
        graph.add_edges_from(line)