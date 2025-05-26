import networkx as nx
from itertools import product

class MaxCutSolver:
    def __call__(self, graph:nx.Graph):
        raise NotImplementedError("MaxCutSolver is an abstract class. Use child class.")
    
class BruteForceMaxCutSolver(MaxCutSolver):
    def __call__(self, graph:nx.Graph):
        nodes = list(graph.nodes())
        n = len(nodes)
        max_cut_value = 0
        best_partition = None
        
        # cut x edges then x + 1 etc. (n times) and select the best value, 2^n
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
    
class GuirobiMaxCutSolver(MaxCutSolver):
    def __call__(self, graph):
        raise NotImplementedError("GuirobiMaxCutSolver not implemented yet.")
    
class GoemansWilliamsonMaxCutSolver(MaxCutSolver):
    def __call__(self, graph):
        raise NotImplementedError("GoemansWilliamsonMaxCutSolver not implemented yet.")
    
class QAOAMaxCutSolver(MaxCutSolver):
    def __call__(self, graph):
        raise NotImplementedError("QAOAMaxCutSolver not implemented yet.")