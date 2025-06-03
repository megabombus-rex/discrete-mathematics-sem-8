import networkx as nx
import numpy as np
import cvxpy as cp
from itertools import product
import gurobipy as gp
from gurobipy import GRB
from constants import GOEMANS_WILLIAMSON_RETRIES_COUNT, QAOA_RETRIES_COUNT, QAOA_DEPTH
import cirq
from cirq.contrib.svg import SVGCircuit
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
        self.depth = QAOA_DEPTH
    
    def estimate_cost(graph, samples):
        """Estimate the cost function of the QAOA on the given graph using the
        provided computational basis bitstrings. No weights used."""
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
    
    def __call__(self, graph):
        node_qubit_map = {node: cirq.LineQubit(node) for node in graph.nodes()}

        alpha = sympy.Symbol('alpha')
        beta = sympy.Symbol('beta')

        qaoa_circuit = cirq.Circuit(
            # Apply Hadamard gates
            cirq.H.on_each(*node_qubit_map.values()),

            # Do ZZ operations between neighbors u, v in the graph. Here, u is a qubit,
            # v is its neighboring qubit, and w is the weight between these qubits.
            (cirq.ZZ(node_qubit_map[u], node_qubit_map[v]) ** (alpha) for (u, v) in graph.edges()),

            # Apply X operations along all nodes of the graph. Again working_graph's
            # nodes are the working_qubits. Note here we use a moment
            # which will force all of the gates into the same line.
            cirq.Moment(cirq.X(qubit) ** beta for qubit in node_qubit_map.values()),

            # All relevant things can be computed in the computational basis.
            (cirq.measure(qubit) for qubit in node_qubit_map.values()),
        )
        
        # Step 3: Optimize angles
        init_params = np.random.uniform(0, np.pi, 2 * p)
        res = minimize(expectation, init_params, bounds=[(0, np.pi)] * 2 * p, method='COBYLA')
        best_params = res.x

        # Step 4: Rebuild and run circuit with optimal params
        circuit = cirq.Circuit()
        circuit.append(cirq.H.on_each(*qubits))

        for layer in range(p):
            gamma = best_params[layer]
            beta = best_params[p + layer]
            for u, v in edges:
                circuit.append(cirq.ZZ(node_qubit_map[u], node_qubit_map[v]) ** (-gamma / np.pi))
            for q in qubits:
                circuit.append(cirq.rx(2 * beta)(q))

        circuit.append(cirq.measure(*qubits, key='result'))

        sim = cirq.Simulator()
        result = sim.run(circuit, repetitions=reps)
        bitstrings = result.measurements['result']

        # Step 5: Analyze best cut
        counts = Counter(tuple(b) for b in bitstrings)
        most_common_bitstring, _ = counts.most_common(1)[0]
        max_cut = bitstring_to_cut(most_common_bitstring)

        # Optional: print solution
        print("Best bitstring:", most_common_bitstring)
        print("Max cut value:", max_cut)

        return most_common_bitstring, max_cut
        
    def bitstring_to_cut(self, bitstring, edges):
            return sum(1 if bitstring[i] != bitstring[j] else 0 for i, j in edges)
        
    # leave the depth at 1 pls
    def expectation(self, params, qubits, edges, depth:int=1):
        gammas = params[:depth]
        betas = params[depth:]

        circuit = cirq.Circuit()
        
        # Initial layer of Hadamards
        circuit.append(cirq.H.on_each(*qubits))

        for layer in range(p):
            gamma = gammas[layer]
            beta = betas[layer]

            # Cost unitary
            for u, v in edges:
                circuit.append(cirq.ZZ(qubits[u], qubits[v]) ** (-gamma / np.pi))

            # Mixer unitary
            for q in qubits:
                circuit.append(cirq.rx(2 * beta)(q))

            # Measure
            circuit.append(cirq.measure(*qubits, key='result'))

            sim = cirq.Simulator()
            result = sim.run(circuit, repetitions=self.repetitions)
            bitstrings = result.measurements['result']
            avg_cut = np.mean([self.bitstring_to_cut(b, edges) for b in bitstrings])
            return -avg_cut  # because we minimize    
        
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
        
    def optimize_AB(self, sim:cirq.Simulator, circuit, graph):
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