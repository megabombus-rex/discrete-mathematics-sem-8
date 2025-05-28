from algorithms import *
import networkx as nx
import time
import csv
import os


class Experiment:         
    def __call__(self, problem_graph:nx.Graph, solver:MaxCutSolver, tries:int=10):
        node_count = len(problem_graph.nodes)
        edge_count = len(problem_graph.edges)
        
        times = []
        solutions = []
        
        if isinstance(solver, BruteForceMaxCutSolver):
            print(f'Is brute solver....')
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
        filename = f"results/{date_str}_{node_count}nodes_{edge_count}edges.csv"
        write_header = not os.path.exists(filename)
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            if write_header:
                writer.writerow(['max_cut_value','time_in_nanoseconds','solver'])
            for i in range(len(solutions)):
                writer.writerow([f'{solutions[i][0]}',f'{times[i]}', solver_name])    
        
        best = max(solutions, key=lambda s: s[0])
        best_str = ', '.join([f"{node}: {group}" for node, group in best[1].items()])
        print(best_str)