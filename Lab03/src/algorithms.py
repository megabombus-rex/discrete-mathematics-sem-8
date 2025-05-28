import networkx as nx
import numpy as np
import cvxpy as cp
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from constants import GOEMANS_WILLIAMSON_RETRIES_COUNT

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
    def __call__(self, graph:nx.Graph):
        model = gp.Model("MaxCut")

        # Binary variables for vertex partitioning
        x = model.addVars(graph.nodes, vtype=GRB.BINARY, name="x")

        # Auxiliary variables for linearization
        y = model.addVars(graph.edges, vtype=GRB.BINARY, name="y")

        # Objective function: maximize total weight of cut edges
        model.setObjective(
            gp.quicksum((x[i] + x[j] - 2 * y[i, j]) for i, j in graph.edges),
            GRB.MAXIMIZE
        )

        # Add McCormick linearization constraints
        for i, j in graph.edges:
            model.addConstr(y[i, j] <= x[i])
            model.addConstr(y[i, j] <= x[j])
            model.addConstr(y[i, j] >= x[i] + x[j] - 1)
            model.addConstr(y[i, j] >= 0)

        model.optimize()

        if model.Status == GRB.OPTIMAL:
            cut_value = model.ObjVal
            partition = {v: int(x[v].X) for v in graph.nodes}
            return cut_value, partition
        else:
            return None, None
    
class GoemansWilliamsonMaxCutSolver(MaxCutSolver):
    def __call__(self, graph:nx.Graph):
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        index = {node: i for i, node in enumerate(nodes)}
        
        W = np.zeros((n, n))
        for u, v, data in graph.edges(data=True):
            i, j = index[u], index[v]
            W[i, j] = W[j, i] = 1
            
        X = cp.Variable((n, n), PSD=True)
        
        # Objective: maximize 0.25 * sum of w_ij * (1 - X_ij)
        objective = cp.Maximize(0.25 * cp.sum(1 - X))
        
        # Constraints: diagonal elements of X must be 1
        constraints = [cp.diag(X) == 1]
        
        # Solve the SDP
        problem = cp.Problem(objective, constraints)
        problem.solve()
        
        # Cholesky-like decomposition of X (rounding step)
        eigvals, eigvecs = np.linalg.eigh(X.value)
        eigvals[eigvals < 0] = 0  # Remove numerical noise
        V = eigvecs @ np.diag(np.sqrt(eigvals))

        best_cut = 0
        best_partition = None

        # Repeat randomized rounding
        for _ in range(GOEMANS_WILLIAMSON_RETRIES_COUNT):
            r = np.random.randn(V.shape[1])
            r /= np.linalg.norm(r)

            assignment = V @ r >= 0
            group = {node: int(assignment[index[node]]) for node in nodes}

            # Count cut value
            cut_value = 0
            for u, v, data in graph.edges(data=True):
                if group[u] != group[v]:
                    cut_value += 1

            if cut_value > best_cut:
                best_cut = cut_value
                best_partition = group

        return best_cut, best_partition
    
class QAOAMaxCutSolver(MaxCutSolver):
    def __call__(self, graph):
        raise NotImplementedError("QAOAMaxCutSolver not implemented yet.")