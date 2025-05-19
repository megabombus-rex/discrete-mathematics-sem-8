import networkx as nx
from itertools import product

class MaxCut:
    def __call__(self, graph:nx.Graph):
        raise NotImplementedError("MaxCut is an abstract class. Use child class.")
    
class BruteForceMaxCut(MaxCut):
    def __call__(self, graph:nx.Graph):
        nodes = list(graph.nodes())
        n = len(nodes)
        max_cut_value = 0
        best_partition = None
        
        for assignment in product([0, 1], repeat = n):
            cut_value = 0
            group = dict(zip(nodes, assignment))
            
            for u, v in graph.edges():
                if group[u] != group[v]:
                    cut_value += 1
            
            if cut_value > max_cut_value:
                max_cut_value = cut_value
                best_partition = group.copy()

        return max_cut_value, best_partition