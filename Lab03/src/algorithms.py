import networkx as nx
import numpy as np
import cvxpy as cp
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from constants import GOEMANS_WILLIAMSON_RETRIES_COUNT, QAOA_RETRIES_COUNT
import cirq
import sympy

class MaxCutSolver:
    def __call__(self, graph:nx.Graph):
        raise NotImplementedError("MaxCutSolver is an abstract class. Use child class.")
    
class BruteForceMaxCutSolver(MaxCutSolver):
    def __call__(self, graph:nx.Graph):
        nodes = list(graph.nodes())
        n = len(nodes)
        max_cut_value = 0
        best_partition = None
        current_try = 0
        
        # cut x edges then x + 1 etc. (n times) and select the best value, 2^n
        for assignment in product([0, 1], repeat = n):
            current_try += 1
            print(f'Current try: {current_try}')
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
    
class QAOACirqMaxCutSolver(MaxCutSolver):
    def __init__(self):
        self.repetitions = QAOA_RETRIES_COUNT
    
    def __call__(self, graph):
        node_qubit_map = {node: cirq.LineQubit(node) for node in graph.nodes()}

        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')

        qaoa_circuit = cirq.Circuit(
            # Apply Hadamard gates for each qubit
            cirq.H.on_each(*node_qubit_map.values()),

            # Do ZZ operations between neighbors u, v in the graph. Here, u is a qubit, and v is its neighboring qubit.
            (cirq.ZZ(node_qubit_map[u], node_qubit_map[v]) ** (alpha) for (u, v) in graph.edges()),

            cirq.Moment(cirq.X(qubit) ** beta for qubit in node_qubit_map.values()),

            # All relevant things can be computed in the computational basis.
            (cirq.measure(qubit, key=str(node)) for node, qubit in node_qubit_map.items()),
        )
        
        sim = cirq.Simulator()
        exp_values, par_values = self.optimize_AB(sim=sim, circuit=qaoa_circuit, graph=graph, alpha=alpha, beta=beta)
        
        # Find the indices of the best (maximum) expectation value
        best_i, best_j = np.unravel_index(np.argmax(exp_values), exp_values.shape)

        # Extract best alpha and beta
        best_alpha, best_beta = par_values[best_i][best_j]

        # Rebuild circuit with best parameters
        final_circuit = cirq.Circuit(
            cirq.H.on_each(*node_qubit_map.values()),
            
            # Use actual numeric values now, not symbols
            (cirq.ZZ(node_qubit_map[u], node_qubit_map[v]) ** best_alpha for (u, v) in graph.edges()),
            
            cirq.Moment(cirq.X(q) ** best_beta for q in node_qubit_map.values()),

            cirq.measure(*node_qubit_map.values(), key='z'))
        
        result = sim.run(final_circuit, repetitions=20000)

        # Convert results into bitstrings
        bitstrings = result.measurements['z']

        best_cut = 0
        best_assignment = None

        for bits in bitstrings:
            current_cut = self.cut_value(bits, graph)
            if current_cut > best_cut:
                best_cut = current_cut
                best_assignment = bits

        partition = {node: bool(bit) for node, bit in zip(graph.nodes(), best_assignment)}

        return best_cut, partition
        
    def bitstring_to_cut(self, bitstring, edges):
            return sum(1 if bitstring[i] != bitstring[j] else 0 for i, j in edges) 
        
    def estimate_cost(self, graph, samples):
        """Estimate the cost function of the QAOA on the given graph using the
        provided computational basis bitstrings."""
        cost_value = 0.0

        # Loop over edge pairs and compute contribution.
        for u, v in graph.edges():
            u_samples = samples[str(u)]
            v_samples = samples[str(v)]

        # Determine if it was a +1 or -1 eigenvalue.
            u_signs = (-1)**u_samples
            v_signs = (-1)**v_samples
            term_signs = u_signs * v_signs

            # Add scaled term to total cost.
            term_val = np.mean(term_signs)
            cost_value += term_val

        return -cost_value    
        
    def optimize_AB(self, sim:cirq.Simulator, circuit, graph, alpha, beta):
        # Set the grid size = number of points in the interval [0, 2π).
        grid_size = 5

        exp_values = np.empty((grid_size, grid_size))
        par_values = np.empty((grid_size, grid_size, 2))

        for i, alpha_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
            for j, beta_value in enumerate(np.linspace(0, 2 * np.pi, grid_size)):
                samples = sim.sample(
                    circuit,
                    params={alpha: alpha_value, beta: beta_value},
                    repetitions=20000
                )
                exp_values[i][j] = self.estimate_cost(graph, samples)
                par_values[i][j] = alpha_value, beta_value
                
        return exp_values, par_values
                
    def cut_value(self, bitstring, graph):
        total = 0
        for u, v in graph.edges():
            if bitstring[u] != bitstring[v]:
                total += 1
        return total