import networkx as nx
import numpy as np
import cvxpy as cp
import scipy.optimize
import scipy.sparse as sp
from openfermion.ops import QubitOperator
from openfermion import get_sparse_operator
from Ansatz import QAOAVariationalAnsatz
import time

# Function to solve Max-Cut using SDP
def solve_max_cut_sdp(graph):
    n = len(graph.nodes)
    X = cp.Variable((n, n), PSD=True)
    constraints = [
        cp.diag(X) == np.ones(n),
        X >> 0
    ]
    objective = cp.Maximize(0.5 * cp.sum([
        graph.edges[(i, j)]['weight'] * (1 - X[i, j])
        for i, j in graph.edges
    ]))
    problem = cp.Problem(objective, constraints)
    optimal_cut_value = problem.solve(solver=cp.SCS)
    np.random.seed(98)
    random_vector = np.random.randn(n)
    approx_cut = np.sign(random_vector @ X.value)
    bitstring = ''.join(['1' if approx_cut[i] > 0 else '0' for i in range(n)])
    return optimal_cut_value, bitstring

# Function to compute the cut value of a solution
def compute_cut_value(graph, bitstring):
    cut_value = 0
    for (u, v, weight) in graph.edges(data='weight'):
        if bitstring[u] != bitstring[v]:
            cut_value += weight
    return cut_value

# Function to compute the approximation ratio
def compute_approximation_ratio(solution_cut_value, optimal_cut_value):
    return solution_cut_value / optimal_cut_value

# QAOA function
def create_reindexed_maxcut_hamiltonian(subgraph):
    reindex_map = {old_index: new_index for new_index, old_index in enumerate(subgraph.nodes)}
    H_C = QubitOperator()
    for u, v, data in subgraph.edges(data=True):
        weight = data['weight']
        u_new, v_new = reindex_map[u], reindex_map[v]
        H_C += weight * 0.5 * (-QubitOperator('') + QubitOperator(f'Z{u_new} Z{v_new}'))
    return H_C

def QAOA(graph, p):
    n = len(graph.nodes)
    hamiltonian = create_reindexed_maxcut_hamiltonian(graph)
    mixer = QubitOperator()  
    for i in range(n):
        mixer += QubitOperator(f'X{i}')
        
    hamiltonian_matrix = get_sparse_operator(hamiltonian, n_qubits=n).tocsc()
    mixer_matrix = get_sparse_operator(mixer, n_qubits=n).tocsc()
    
    reference_state = sp.csc_matrix(np.full((2**n, 1), 1/np.sqrt(2**n)), dtype=np.complex128)
    
    parameters = np.random.rand(2 * p)
    
    trial_model = QAOAVariationalAnsatz(hamiltonian_matrix, mixer_matrix, reference_state, parameters)
    
    opt_result = scipy.optimize.minimize(trial_model.energy, parameters, method='Nelder-Mead', callback=trial_model.callback)

    optimized_parameters = opt_result.x
    final_energy = trial_model.energy(optimized_parameters)
    
    final_state = trial_model.prepare_state(optimized_parameters)
    probabilities = np.abs(final_state.toarray().flatten())**2
    max_prob_index = np.argmax(probabilities)
    solution_bitstring = format(max_prob_index, f'0{n}b')

    print("Optimized Parameters:", optimized_parameters)
    print("Final Energy:", final_energy)
    print("Solution Bitstring:", solution_bitstring)

    return final_energy, solution_bitstring

# Function to show graph partitions
def show_graph_partitions(subgraphs, level, file):
    file.write(f"Level {level} partitions:\n")
    for i, subgraph in enumerate(subgraphs):
        file.write(f"  Subgraph {i}: Nodes {list(subgraph.nodes)}\n")

# Function to create high-level graph
def create_high_level_graph(graph, subgraphs, subgraph_solutions, file, level):
    high_level_graph = nx.Graph()
    for i, subgraph_i in enumerate(subgraphs):
        for j, subgraph_j in enumerate(subgraphs):
            if i != j:
                weight = 0
                for u in subgraph_i:
                    for v in subgraph_j:
                        if graph.has_edge(u, v):
                            bit_u = subgraph_solutions[i][list(subgraph_i.nodes).index(u)]
                            bit_v = subgraph_solutions[j][list(subgraph_j.nodes).index(v)]
                            weight += 1 if bit_u == bit_v else -1
                high_level_graph.add_edge(i, j, weight=weight)
    
    file.write(f"High-level graph (Level {level}) edges: {list(high_level_graph.edges(data=True))}\n")
    return high_level_graph

# Function to merge solutions
def merge_solutions(level_bitstring, subgraph_solutions):
    final_solution = []
    for i, bit in enumerate(level_bitstring):
        subgraph_solution = subgraph_solutions[i]
        if bit == 1:
            subgraph_solution = 1 - np.array(subgraph_solution)  
        final_solution.extend(subgraph_solution)
    return final_solution

# Function to apply QAOA² recursively
def QAOA_squared(graph, max_size, p=1, file=None, current_level=1, partition_method=None):
    subgraphs, levels = partition_method(graph, max_size, file, current_level)
    show_graph_partitions(subgraphs, current_level, file)
    file.write(f"Number of hierarchical levels: {levels}\n")

    subgraph_solutions = []
    node_mapping = []
    for subgraph in subgraphs:
        _, bitstring = QAOA(subgraph, p)
        subgraph_solutions.append([int(bit) for bit in bitstring])
        node_mapping.append(list(subgraph.nodes))
        file.write(f"Subgraph solution: {bitstring}\n")

    if len(subgraphs) <= max_size:
        high_level_graph = create_high_level_graph(graph, subgraphs, subgraph_solutions, file, current_level + 1)
        _, high_level_bitstring = QAOA(high_level_graph, p)
        file.write(f"High-level solution bitstring: {high_level_bitstring}\n")
        final_solution = merge_solutions([int(bit) for bit in high_level_bitstring], subgraph_solutions)
    else:
        high_level_graph = create_high_level_graph(graph, subgraphs, subgraph_solutions, file, current_level + 1)
        final_solution, _ = QAOA_squared(high_level_graph, max_size, p, file, current_level + 1, partition_method)

    original_solution = np.zeros(len(graph.nodes), dtype=int)
    for i, bitstring in enumerate(subgraph_solutions):
        for j, node in enumerate(node_mapping[i]):
            original_solution[node] = bitstring[j]

    return original_solution, levels

# Function to run QAOA² and evaluate performance
def qaoa_performance(graph, max_size, p, filename, partition_method):
    with open(filename, "w") as file:
        start_time = time.time()
        final_solution, levels = QAOA_squared(graph, max_size, p, file, partition_method=partition_method)
        end_time = time.time()
        
        solution_cut_value = compute_cut_value(graph, final_solution)
        optimal_cut_value, _ = solve_max_cut_sdp(graph)
        approximation_ratio = compute_approximation_ratio(solution_cut_value, optimal_cut_value)
        
        execution_time = end_time - start_time
        
        file.write(f"Final merged solution: {final_solution}\n")
        file.write(f"Cut value of the solution: {solution_cut_value}\n")
        file.write(f"Optimal cut value (SDP): {optimal_cut_value}\n")
        file.write(f"Approximation ratio: {approximation_ratio}\n")
        file.write(f"Execution time: {execution_time} seconds\n")
        file.write(f"Number of hierarchical levels: {levels}\n")

