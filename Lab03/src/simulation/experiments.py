from algorithms import *
import networkx as nx
import time
import csv
import os


class Experiment:         
    def __call__(self, problem_graph:nx.Graph, graph_name:str, solver:MaxCutSolver, tries:int=10):
        node_count = len(problem_graph.nodes)
        edge_count = len(problem_graph.edges)
        
        times = []
        solutions = []
        
        if isinstance(solver, BruteForceMaxCutSolver) or isinstance(solver, GuirobiMaxCutSolver) :
            print(f'Is brute solver.... or gurobi....')
            start = time.perf_counter_ns()
            (max_value, partition) = solver(problem_graph)
            end = time.perf_counter_ns()
            duration_ns = end - start
            times.append(duration_ns)
            solutions.append((max_value, partition))
        else:
            for i in range(tries):
                print(f'Test {i + 1}')
                start = time.perf_counter_ns()
                (max_value, partition) = solver(problem_graph)
                end = time.perf_counter_ns()
                duration_ns = end - start
                times.append(duration_ns)
                solutions.append((max_value, partition))
                                   
        solver_name = type(solver).__name__
                
        date_str = time.strftime("%Y-%m-%d")
        filename = f"results/{date_str}_results.csv"
        write_header = not os.path.exists(filename)
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['max_cut_value','time_in_nanoseconds','solver', 'node_count', 'edge_count', 'graph_name'])
            for i in range(len(solutions)):
                writer.writerow([f'{solutions[i][0]}',f'{times[i]}', solver_name, node_count, edge_count, graph_name])    
        
        best = max(solutions, key=lambda s: s[0])
        best_str = ', '.join([f"{node}: {group}" for node, group in best[1].items()])
        print(best_str)