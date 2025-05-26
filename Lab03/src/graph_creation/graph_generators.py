import networkx as nx
import numpy as np
import math
import random

class GraphGenerator():
    @classmethod
    def generate_simple_graph_temperature(self, size, temperature):
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(0,size,1))
        
        line = [(i - 1, i) for i in range(1, size - 1)]
        
        max_edges = int(math.floor(temperature * size))
        
        #edges = [(random.randint(0, size), random.randint(0, size)) for i in max_edges]
        
        additional_edges = []
        for i in range(max_edges):
            edge = (random.randint(0, size), random.randint(0, size))
            if not edge in additional_edges:
                additional_edges.append(edge)
        
        line += additional_edges
        
        for i in range(len(line)):
            print(f'Edge between: {line[i]} ')
        
        graph.add_edges_from(line)
        return graph