import matplotlib.pyplot as plt
import networkx as nx

class Painter:
    @classmethod
    def visualize_cut(self, G, partition):
        pos = nx.spring_layout(G, seed=42)  # For consistent layout

        group_colors = ['red', 'blue']
        node_colors = [group_colors[partition[node]] for node in G.nodes()]

        cut_edges = []
        same_group_edges = []
        for u, v in G.edges():
            if partition[u] != partition[v]:
                cut_edges.append((u, v))
            else:
                same_group_edges.append((u, v))

        plt.figure(figsize=(8, 6))
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, edgecolors='black')
        nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

        nx.draw_networkx_edges(G, pos, edgelist=same_group_edges, edge_color='blue', width=1)
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', style='dashed', width=1)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Max-Cut Partition (Red = Cut Edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    @classmethod    
    def visualize_graph(self, G:nx.Graph, title="Graph Visualization"):
        pos = {}
        
        stack_horizontal_size = 30
        i = 0
        j = 0
        for node in G.nodes:
            pos[node] = (i % stack_horizontal_size, j)
            print(f'Position: {pos[node]}')
            if i + 1 >= stack_horizontal_size:
                j += 1 
            i = (i + 1) % stack_horizontal_size
            
        # Draw the graph
        plt.figure(figsize=(10, 6))
        nx.draw(G, pos, node_color='black', node_size=20, edge_color='green', linewidths='0.1')

        plt.title(title)
        plt.axis('off')
        #plt.tight_layout()
        plt.show()
