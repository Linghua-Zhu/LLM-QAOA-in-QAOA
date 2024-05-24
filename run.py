import networkx as nx
from QAOA_in_QAOA import qaoa_performance
from partition_methods import (
    partition_graph_louvain,
    partition_graph_asyn_fluidc,
    partition_graph_random,
    partition_graph_girvan_newman,
    partition_graph_spectral
)

def main():
    # Define the graph
    G = nx.Graph()
    edges_with_weights = [
        (0, 1, 0.5), (0, 2, 0.3), (0, 3, 0.77), (1, 2, 0.67), (1, 3, 0.99), (1, 7, 0.54), (2, 4, 0.33), (3, 10, 0.35), 
        (4, 5, 0.3), (4, 6, 0.9), (4, 7, 0.10), (5, 6, 0.48), (5, 7, 0.23), (5, 17, 0.21), (6, 7, 0.78), (6, 8, 0.60),
        (6, 24, 0.22), (7, 10, 0.54), (8, 9, 0.35), (9, 11, 0.33), (9, 17, 0.52), (9, 26, 0.12), (10, 11, 0.24), 
        (10, 12, 0.44), (11, 12, 0.09), (13, 14, 0.19), (13, 17, 0.22), (14, 15, 0.21), (14, 16, 0.08), (14, 17, 0.32),
        (15, 18, 0.32), (16, 17, 0.42), (16, 18, 0.78), (16, 21, 0.36), (16, 24, 0.66), (17, 25, 0.76), (18, 19, 0.02),
        (18, 20, 0.57), (19, 20, 0.65), (20, 21, 0.39), (20, 22, 0.91), (21, 22, 0.43), (21, 23, 0.57), (21, 25, 0.45), 
        (22, 23, 0.10), (23, 26, 0.38), (24, 25, 0.66), (24, 26, 0.67), (25, 26, 0.39)
    ]
    G.add_weighted_edges_from(edges_with_weights)

    # Define parameters
    max_size = 4
    p = 1
    output_filename = "9.txt"

    # Choose partition method: 'asyn_fluidc', 'random', 'girvan_newman', or 'spectral'
    partition_method = 'girvan_newman'  # or 'asyn_fluidc', 'random', 'girvan_newman', 'spectral'

    partition_methods = {
        'louvain': partition_graph_louvain,
        'asyn_fluidc': partition_graph_asyn_fluidc,
        'random': partition_graph_random,
        'girvan_newman': partition_graph_girvan_newman,
        'spectral': partition_graph_spectral
    }

    qaoa_performance(G, max_size, p, output_filename, partition_methods[partition_method])

if __name__ == "__main__":
    main()
