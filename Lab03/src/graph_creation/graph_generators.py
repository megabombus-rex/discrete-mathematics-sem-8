import networkx as nx
import numpy as np
import math
import random
import time
import os
import csv

class GraphGenerator():
    @classmethod
    def generate_simple_graph_temperature(self, size, temperature):
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(0,size - 1))
        
        line = [(i - 1, i) for i in range(1, size - 1)]
        
        max_edges = int(math.floor(temperature * size))
        
        additional_edges = []
        for i in range(max_edges):
            edge = (random.randint(0, size), random.randint(0, size))
            if edge not in additional_edges and edge[0] != edge[1]:
                additional_edges.append(edge)
        
        line += additional_edges
        
        graph.add_edges_from(line)
        
        node_count = len(graph.nodes)
        edge_count = len(graph.edges)
        
        date_str = time.strftime("%Y-%m-%d-%H-%M-%S") + f'{node_count}_{edge_count}'
        
        filename = f"data/set6/graph_{date_str}.csv"
        write_header = not os.path.exists(filename)
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file, delimiter=' ')
            if write_header:
                writer.writerow([node_count, edge_count])
            for u, v in graph.edges():
                writer.writerow([u, v])
        
        return graph