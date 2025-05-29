"""
Simulation utilities for the Kuramoto Model Simulator.
Contains simulation runner and helper functions.
"""

import streamlit as st
import numpy as np
import json
from src.models.kuramoto_model import KuramotoModel


@st.cache_data(ttl=300)
def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step=None, random_seed=None, 
                  adjacency_matrix=None):
    """
    Run a Kuramoto model simulation with the specified parameters and return the results.
    
    Parameters:
    -----------
    n_oscillators : int
        Number of oscillators
    coupling_strength : float
        Coupling strength parameter K
    frequencies : ndarray
        Natural frequencies of oscillators
    simulation_time : float
        Total simulation time
    time_step : float, optional
        DEPRECATED: This parameter is no longer used. The time step is now automatically calculated
        based on oscillator frequencies to ensure numerical stability and accuracy.
        The parameter is kept for backward compatibility with saved configurations.
    random_seed : int, optional
        Seed for random number generation
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix defining network connectivity
        
    Returns:
    --------
    tuple
        (model, times, phases, order_parameter)
    """
    # Convert random_seed to integer to prevent type errors
    if random_seed is not None:
        random_seed = int(random_seed)
    
    # Initialize the model with given parameters
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        adjacency_matrix=adjacency_matrix,
        simulation_time=simulation_time,
        random_seed=random_seed
    )
    
    # Run the simulation
    times, phases, order_parameter = model.simulate()
    
    # Print simulation results for debugging
    print("Simulation successful!")
    print(f"  times shape: {times.shape}")
    print(f"  phases shape: {phases.shape}")
    print(f"  order_parameter shape: {order_parameter.shape}")
    
    return model, times, phases, order_parameter


def parse_json_parameters(json_string):
    """
    Parse JSON string for frequency distribution parameters.
    
    Parameters:
    -----------
    json_string : str
        JSON string containing parameters
        
    Returns:
    --------
    dict or None
        Parsed parameters or None if parsing fails
    """
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        return None


def generate_frequencies(n_oscillators, distribution_type, parameters=None):
    """
    Generate natural frequencies based on distribution type and parameters.
    
    Parameters:
    -----------
    n_oscillators : int
        Number of oscillators
    distribution_type : str
        Type of distribution ('Normal', 'Uniform', 'Bimodal', 'Custom')
    parameters : dict, optional
        Distribution parameters
        
    Returns:
    --------
    tuple
        (frequencies, freq_params_dict)
    """
    if parameters is None:
        parameters = {}
    
    if distribution_type == "Normal":
        mean = parameters.get('mean', 0.0)
        std = parameters.get('std', 1.0)
        frequencies = np.random.normal(mean, std, n_oscillators)
        freq_params = {'mean': mean, 'std': std}
        
    elif distribution_type == "Uniform":
        low = parameters.get('low', -1.0)
        high = parameters.get('high', 1.0)
        frequencies = np.random.uniform(low, high, n_oscillators)
        freq_params = {'low': low, 'high': high}
        
    elif distribution_type == "Bimodal":
        mean1 = parameters.get('mean1', -1.0)
        mean2 = parameters.get('mean2', 1.0)
        std1 = parameters.get('std1', 0.2)
        std2 = parameters.get('std2', 0.2)
        weight = parameters.get('weight', 0.5)
        
        n1 = int(n_oscillators * weight)
        n2 = n_oscillators - n1
        
        freq1 = np.random.normal(mean1, std1, n1)
        freq2 = np.random.normal(mean2, std2, n2)
        frequencies = np.concatenate([freq1, freq2])
        np.random.shuffle(frequencies)
        
        freq_params = {
            'mean1': mean1, 'mean2': mean2,
            'std1': std1, 'std2': std2,
            'weight': weight
        }
        
    else:  # Custom or default
        frequencies = np.random.normal(0, 1, n_oscillators)
        freq_params = {'mean': 0.0, 'std': 1.0}
    
    return frequencies, freq_params


def create_adjacency_matrix(n_oscillators, network_type, **kwargs):
    """
    Create adjacency matrix based on network type.
    
    Parameters:
    -----------
    n_oscillators : int
        Number of oscillators
    network_type : str
        Type of network
    **kwargs : dict
        Additional parameters for network generation
        
    Returns:
    --------
    ndarray
        Adjacency matrix
    """
    if network_type == "All-to-All":
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(adj_matrix, 0)
        
    elif network_type == "Random":
        connection_probability = kwargs.get('connection_probability', 0.3)
        adj_matrix = np.random.random((n_oscillators, n_oscillators)) < connection_probability
        adj_matrix = adj_matrix.astype(float)
        np.fill_diagonal(adj_matrix, 0)
        # Make symmetric
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0).astype(float)
        
    elif network_type == "Small World":
        k = kwargs.get('k', 4)  # Number of nearest neighbors
        p = kwargs.get('p', 0.1)  # Rewiring probability
        
        # Create ring lattice
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            for j in range(1, k//2 + 1):
                neighbor1 = (i + j) % n_oscillators
                neighbor2 = (i - j) % n_oscillators
                adj_matrix[i, neighbor1] = 1
                adj_matrix[i, neighbor2] = 1
        
        # Rewire with probability p
        for i in range(n_oscillators):
            for j in range(i+1, n_oscillators):
                if adj_matrix[i, j] == 1 and np.random.random() < p:
                    # Remove edge
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0
                    # Add random edge
                    new_j = np.random.choice([k for k in range(n_oscillators) if k != i and adj_matrix[i, k] == 0])
                    adj_matrix[i, new_j] = 1
                    adj_matrix[new_j, i] = 1
    
    elif network_type == "Scale-Free":
        # Simple preferential attachment model
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        m = kwargs.get('m', 2)  # Number of edges to attach from new node
        
        # Start with a small complete graph
        initial_nodes = min(m + 1, n_oscillators)
        for i in range(initial_nodes):
            for j in range(i+1, initial_nodes):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Add remaining nodes with preferential attachment
        for new_node in range(initial_nodes, n_oscillators):
            degrees = np.sum(adj_matrix, axis=0)
            degrees[new_node] = 0  # New node has no connections yet
            
            if np.sum(degrees) == 0:
                # If no connections exist, connect to first node
                target = 0
                adj_matrix[new_node, target] = 1
                adj_matrix[target, new_node] = 1
            else:
                # Preferential attachment
                probabilities = degrees / np.sum(degrees)
                targets = np.random.choice(new_node, size=min(m, new_node), 
                                         replace=False, p=probabilities[:new_node])
                
                for target in targets:
                    adj_matrix[new_node, target] = 1
                    adj_matrix[target, new_node] = 1
    
    else:  # Default to random
        adj_matrix = np.random.random((n_oscillators, n_oscillators)) < 0.3
        adj_matrix = adj_matrix.astype(float)
        np.fill_diagonal(adj_matrix, 0)
        adj_matrix = (adj_matrix + adj_matrix.T) / 2
        adj_matrix = (adj_matrix > 0).astype(float)
    
    return adj_matrix


def get_network_statistics(adjacency_matrix):
    """
    Calculate basic network statistics.
    
    Parameters:
    -----------
    adjacency_matrix : ndarray
        Network adjacency matrix
        
    Returns:
    --------
    dict
        Dictionary containing network statistics
    """
    n_nodes = adjacency_matrix.shape[0]
    n_edges = int(np.sum(adjacency_matrix) / 2)  # Divide by 2 for undirected graph
    
    # Calculate degree statistics
    degrees = np.sum(adjacency_matrix, axis=0)
    avg_degree = np.mean(degrees)
    max_degree = np.max(degrees)
    min_degree = np.min(degrees)
    
    # Calculate density
    max_possible_edges = n_nodes * (n_nodes - 1) / 2
    density = n_edges / max_possible_edges if max_possible_edges > 0 else 0
    
    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'density': density,
        'avg_degree': avg_degree,
        'max_degree': max_degree,
        'min_degree': min_degree
    }