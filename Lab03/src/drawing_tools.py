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

        nx.draw_networkx_edges(G, pos, edgelist=same_group_edges, style='dashed', edge_color='gray')
        nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', width=2)

        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

        plt.title("Max-Cut Partition (Red = Cut Edges)")
        plt.axis('off')
        plt.tight_layout()
        plt.show()
