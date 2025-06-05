import csv

def read_results_csv(self, path):
    with open(path, "r") as file:
            first_line = file.readline()
            
            node_count = first_line.strip().split(' ')[0]
            edge_list = []
                    
            while first_line:
                first_line = file.readline()
                if not first_line:
                    continue
                encoded_edge = first_line.strip().split(' ')
                #G.add_edge(int(encoded_edge[0].strip()), int(encoded_edge[1].strip()))
                print(f'Added edge ({encoded_edge[0]}, {encoded_edge[1]})')
    pass