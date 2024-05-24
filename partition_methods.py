import networkx as nx
import numpy as np
from sklearn.cluster import SpectralClustering

def show_graph_partitions(subgraphs, level, file):
    file.write(f"Level {level} partitions:\n")
    for i, subgraph in enumerate(subgraphs):
        file.write(f"  Subgraph {i}: Nodes {list(subgraph.nodes)}\n")

def partition_graph_louvain(graph, max_size, file, current_level=1):
    level_count = current_level
    subgraphs = [graph]
    
    while any(len(subgraph) > max_size for subgraph in subgraphs):
        new_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph) > max_size:
                partition = nx.algorithms.community.louvain_communities(subgraph, seed=42)
                for community in partition:
                    sub_subgraph = subgraph.subgraph(community).copy()
                    if len(sub_subgraph) > max_size:
                        new_subgraphs.extend(partition_graph_louvain(sub_subgraph, max_size, file, level_count + 1)[0])
                    else:
                        new_subgraphs.append(sub_subgraph)
            else:
                new_subgraphs.append(subgraph)
        
        subgraphs = new_subgraphs
        level_count += 1
        show_graph_partitions(subgraphs, level_count, file)
    
    return subgraphs, level_count

def partition_graph_asyn_fluidc(graph, max_size, file, current_level=1):
    level_count = current_level
    subgraphs = [graph]
    
    while any(len(subgraph) > max_size for subgraph in subgraphs):
        new_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph) > max_size:
                communities = list(nx.algorithms.community.asyn_fluidc(subgraph, 2))
                for community in communities:
                    sub_subgraph = subgraph.subgraph(community).copy()
                    if len(sub_subgraph) > max_size:
                        new_subgraphs.extend(partition_graph_asyn_fluidc(sub_subgraph, max_size, file, level_count + 1)[0])
                    else:
                        new_subgraphs.append(sub_subgraph)
            else:
                new_subgraphs.append(subgraph)
        
        subgraphs = new_subgraphs
        level_count += 1
        show_graph_partitions(subgraphs, level_count, file)
    
    return subgraphs, level_count

def partition_graph_random(graph, max_size, file, current_level=1):
    level_count = current_level
    subgraphs = [graph]
    
    while any(len(subgraph) > max_size for subgraph in subgraphs):
        new_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph) > max_size:
                nodes = list(subgraph.nodes)
                np.random.shuffle(nodes)
                for i in range(0, len(nodes), max_size):
                    sub_subgraph = subgraph.subgraph(nodes[i:i + max_size]).copy()
                    new_subgraphs.append(sub_subgraph)
            else:
                new_subgraphs.append(subgraph)
        
        subgraphs = new_subgraphs
        level_count += 1
        show_graph_partitions(subgraphs, level_count, file)
    
    return subgraphs, level_count

def partition_graph_girvan_newman(graph, max_size, file, current_level=1):
    level_count = current_level
    subgraphs = [graph]
    
    while any(len(subgraph) > max_size for subgraph in subgraphs):
        new_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph) > max_size:
                comp = nx.algorithms.community.girvan_newman(subgraph)
                limited = tuple(sorted(c) for c in next(comp))
                for community in limited:
                    sub_subgraph = subgraph.subgraph(community).copy()
                    if len(sub_subgraph) > max_size:
                        new_subgraphs.extend(partition_graph_girvan_newman(sub_subgraph, max_size, file, level_count + 1)[0])
                    else:
                        new_subgraphs.append(sub_subgraph)
            else:
                new_subgraphs.append(subgraph)
        
        subgraphs = new_subgraphs
        level_count += 1
        show_graph_partitions(subgraphs, level_count, file)
    
    return subgraphs, level_count

def partition_graph_spectral(graph, max_size, file, current_level=1):
    level_count = current_level
    subgraphs = [graph]
    
    while any(len(subgraph) > max_size for subgraph in subgraphs):
        new_subgraphs = []
        for subgraph in subgraphs:
            if len(subgraph) > max_size:
                adjacency_matrix = nx.to_numpy_array(subgraph)
                clustering = SpectralClustering(n_clusters=2, affinity='precomputed').fit(adjacency_matrix)
                labels = clustering.labels_
                nodes = list(subgraph.nodes)
                community_1 = [nodes[i] for i in range(len(nodes)) if labels[i] == 0]
                community_2 = [nodes[i] for i in range(len(nodes)) if labels[i] == 1]
                
                for community in [community_1, community_2]:
                    sub_subgraph = subgraph.subgraph(community).copy()
                    if len(sub_subgraph) > max_size:
                        new_subgraphs.extend(partition_graph_spectral(sub_subgraph, max_size, file, level_count + 1)[0])
                    else:
                        new_subgraphs.append(sub_subgraph)
            else:
                new_subgraphs.append(subgraph)
        
        subgraphs = new_subgraphs
        level_count += 1
        show_graph_partitions(subgraphs, level_count, file)
    
    return subgraphs, level_count
