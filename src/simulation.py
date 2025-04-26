"""
Core simulation functionality for the Kuramoto Model Simulator.
"""

import numpy as np
from src.models.kuramoto_model import KuramotoModel

def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, random_seed=None, 
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
    random_seed : int, optional
        Seed for random number generation
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix defining network connectivity
        
    Returns:
    --------
    tuple
        (model, times, phases, order_parameter)
    """
    # Create the Kuramoto model with given parameters
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        adjacency_matrix=adjacency_matrix,
        simulation_time=simulation_time,
        random_seed=random_seed
    )
    
    # Run the simulation - the model will automatically calculate optimal step size
    times, phases, order_parameter = model.simulate()
    
    print("Simulation successful!")
    print(f"  times shape: {times.shape}")
    print(f"  phases shape: {phases.shape}")
    print(f"  order_parameter shape: {order_parameter.shape}")
    
    return model, times, phases, order_parameter