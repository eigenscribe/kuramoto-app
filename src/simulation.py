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
    # Set random seed if provided
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Initial phases randomly distributed in [0, 2π)
    initial_phases = np.random.uniform(0, 2*np.pi, n_oscillators)
    
    # Create the Kuramoto model with given parameters
    model = KuramotoModel(
        frequencies=frequencies,
        coupling_strength=coupling_strength,
        initial_phases=initial_phases,
        adjacency_matrix=adjacency_matrix
    )
    
    # Determine simulation parameters
    # The time step is automatically calculated based on the highest oscillator frequency
    # to ensure numerical stability and accuracy
    # Calculate optimal step size for numerical stability
    # The step size is inversely proportional to the highest frequency component
    omega_max = np.max(np.abs(frequencies))
    
    # Add coupling effect to the max frequency for conservative step size
    # The eigenvalues of the system are shifted by the coupling effect
    lambda_max = omega_max + coupling_strength
    
    # Use 1/34 of the max frequency period as step size for good accuracy
    # This is a conservative estimate that works well for most cases
    max_step = 1.0 / (34 * lambda_max) if lambda_max > 0 else 0.03
    
    # Cap the maximum step size to prevent extremely slow oscillators from causing issues
    max_step = min(max_step, 0.03)
    
    # Print step size info for debugging
    print(f"Simulating with max_step={max_step:.5f}, ω_max={omega_max:.5f}, λ_max={lambda_max:.5f}")
    
    # We want a reasonable number of time points to visualize the dynamics
    # Use a variable number of time points based on the duration:
    # - At least 500 points to ensure smooth visualization
    # - At most 2000 points to prevent performance issues
    # - Approximately 50 points per time unit is a good balance
    # This gives us smooth visualization for large time ranges without excessive points
    n_points = min(max(500, int(50 * simulation_time)), 2000)
    
    # Create a uniform time array for visualization
    t_eval = np.linspace(0, simulation_time, n_points)
    print(f"Using t_eval with {len(t_eval)} points for simulation duration {simulation_time}")
    
    # Run the simulation with automatic step size selection
    times, phases = model.simulate(
        t_span=[0, simulation_time],
        t_eval=t_eval
    )
    
    # Calculate the order parameter (level of synchronization)
    order_parameter = model.calculate_order_parameter(phases)
    
    print("Simulation successful!")
    print(f"  times shape: {times.shape}")
    print(f"  phases shape: {phases.shape}")
    print(f"  order_parameter shape: {order_parameter.shape}")
    
    return model, times, phases, order_parameter