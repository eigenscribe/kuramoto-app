"""
Helper module for working with Kuramoto model simulations for machine learning.

This module provides utilities and examples for:
1. Creating datasets of Kuramoto simulations with varied parameters
2. Extracting relevant features for ML tasks
3. Exporting datasets in formats suitable for different ML frameworks
4. Preparing data for Lagrangian Neural Networks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.utils.database import (
    create_ml_dataset,
    add_simulation_to_dataset,
    extract_features,
    get_ml_dataset,
    export_ml_dataset,
    run_batch_simulations,
    list_ml_datasets
)
from src.models.kuramoto_model import KuramotoModel


def generate_parameter_sweep(param_grid):
    """
    Generate a list of parameter combinations from a parameter grid.
    
    Parameters:
    -----------
    param_grid : dict
        Dictionary where keys are parameter names and values are lists of parameter values
        
    Returns:
    --------
    list
        List of dictionaries, each containing one combination of parameters
    """
    import itertools
    
    # Get all parameter names and possible values
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    # Generate all combinations of parameter values
    combinations = list(itertools.product(*param_values))
    
    # Create a list of parameter dictionaries
    param_dicts = []
    for combo in combinations:
        param_dict = {name: value for name, value in zip(param_names, combo)}
        param_dicts.append(param_dict)
    
    return param_dicts


def create_critical_coupling_dataset(name="critical_coupling_dataset", n_oscillators=10, 
                                   frequency_std=0.1, k_values=None):
    """
    Create a dataset focusing on the transition to synchronization at critical coupling strength.
    
    Parameters:
    -----------
    name : str
        Name for the new dataset
    n_oscillators : int
        Number of oscillators in each simulation
    frequency_std : float
        Standard deviation of the natural frequencies
    k_values : list, optional
        List of coupling strength values to use. If None, will use a range around the critical point.
        
    Returns:
    --------
    int
        ID of the created dataset
    """
    # Calculate theoretical critical coupling strength
    # For Kuramoto with normal distribution of frequencies, it's approximately 2*std/π
    critical_k = 2 * frequency_std / np.pi
    
    # Create a range of K values around the critical point if not provided
    if k_values is None:
        k_values = np.linspace(0.2 * critical_k, 3 * critical_k, 30)
    
    # Create the parameter variations
    config_variations = []
    for k in k_values:
        config_variations.append({
            'coupling_strength': k,
            'frequency_distribution': 'normal',
            'frequency_params': {'mean': 0.0, 'std': frequency_std}
        })
    
    # Base configuration
    base_config = {
        'n_oscillators': n_oscillators,
        'simulation_time': 50.0,  # Longer time to ensure steady state
        'time_step': 0.05,
        'random_seed': 42,
        'network_type': 'all-to-all'
    }
    
    # Run the batch simulations and create dataset
    batch_result = run_batch_simulations(
        config_variations=config_variations,
        base_config=base_config,
        dataset_name=name
    )
    
    # Extract relevant features
    dataset_id = batch_result['dataset_id']
    feature_config = {
        'steady_state_sync': {
            'type': 'target',
            'description': 'Final synchronization level (order parameter at t_max)',
            'extraction': 'order_parameter.r[-1]'
        },
        'order_parameter_timeseries': {
            'type': 'input',
            'description': 'Order parameter r(t) over time',
            'extraction': 'order_parameter.r'
        },
        'coupling_strength': {
            'type': 'input',
            'description': 'Coupling strength (K) value',
            'extraction': 'params'
        },
        'frequencies': {
            'type': 'input',
            'description': 'Natural frequencies of oscillators',
            'extraction': 'frequencies'
        }
    }
    
    extract_features(dataset_id, feature_config)
    
    print(f"Created dataset '{name}' (ID: {dataset_id}) with {len(config_variations)} simulations")
    print(f"Theoretical critical coupling: {critical_k:.4f}")
    
    return dataset_id


def create_network_topology_dataset(name="network_topology_dataset", n_oscillators=10,
                                  coupling_strength=1.0, n_samples=20):
    """
    Create a dataset comparing different network topologies for the same oscillator parameters.
    
    Parameters:
    -----------
    name : str
        Name for the new dataset
    n_oscillators : int
        Number of oscillators in each simulation
    coupling_strength : float
        Coupling strength to use for all simulations
    n_samples : int
        Number of random samples to generate for each network type
        
    Returns:
    --------
    int
        ID of the created dataset
    """
    # Create the parameter variations
    config_variations = []
    
    # Multiple samples for each network type with different random seeds
    for network_type in ['all-to-all', 'ring', 'random']:
        for i in range(n_samples):
            config_variations.append({
                'network_type': network_type,
                'random_seed': 1000 + i,
                'frequency_distribution': 'normal',
                'frequency_params': {'mean': 0.0, 'std': a0.1}
            })
    
    # Base configuration
    base_config = {
        'n_oscillators': n_oscillators,
        'coupling_strength': coupling_strength,
        'simulation_time': 20.0,
        'time_step': 0.05
    }
    
    # Run the batch simulations and create dataset
    batch_result = run_batch_simulations(
        config_variations=config_variations,
        base_config=base_config,
        dataset_name=name
    )
    
    # Extract relevant features
    dataset_id = batch_result['dataset_id']
    feature_config = {
        'network_type': {
            'type': 'target',
            'description': 'Type of network topology',
            'extraction': 'params'
        },
        'adjacency_matrix': {
            'type': 'input',
            'description': 'Network adjacency matrix',
            'extraction': 'adjacency_matrix'
        },
        'sync_trajectory': {
            'type': 'input',
            'description': 'Synchronization trajectory r(t)',
            'extraction': 'order_parameter.r'
        },
        'final_phases': {
            'type': 'input',
            'description': 'Final phase configuration',
            'extraction': 'phases'
        }
    }
    
    extract_features(dataset_id, feature_config)
    
    print(f"Created dataset '{name}' (ID: {dataset_id}) with {len(config_variations)} simulations")
    print(f"Network types: all-to-all, ring, random with {n_samples} samples each")
    
    return dataset_id


def visualize_dataset(dataset_id):
    """
    Visualize the key features of a machine learning dataset.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset to visualize
        
    Returns:
    --------
    None
    """
    # Load the dataset
    dataset = get_ml_dataset(dataset_id)
    if not dataset:
        print(f"Dataset with ID {dataset_id} not found")
        return
    
    print(f"Dataset: {dataset['name']}")
    print(f"Description: {dataset['description']}")
    print(f"Simulations: {len(dataset['simulations'])}")
    print(f"Features: {len(dataset['features'])}")
    
    # Extract feature data by name
    feature_data = {}
    for feature in dataset['features']:
        feature_data[feature['name']] = feature
    
    # Create visualizations based on dataset type and features
    if dataset['feature_type'] == 'time_series':
        # Plot order parameter time series if available
        if 'order_parameter_timeseries' in feature_data:
            plt.figure(figsize=(10, 6))
            r_values = feature_data['order_parameter_timeseries']['data']
            
            # If we have coupling strengths, color by K
            if 'coupling_strength' in feature_data:
                k_values = feature_data['coupling_strength']['data']['value']
                
                # Sort by coupling strength
                sorted_indices = np.argsort(k_values)
                k_values = k_values[sorted_indices]
                
                # Plot each trajectory colored by K value
                cmap = plt.cm.viridis
                for i, idx in enumerate(sorted_indices):
                    color = cmap(i / len(sorted_indices))
                    plt.plot(r_values[idx], color=color, alpha=0.7)
                
                # Add colorbar
                from matplotlib.cm import ScalarMappable
                sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(k_values.min(), k_values.max()))
                sm.set_array([])
                cbar = plt.colorbar(sm)
                cbar.set_label('Coupling Strength (K)')
            else:
                # Plot without color coding
                for i in range(len(r_values)):
                    plt.plot(r_values[i], alpha=0.5)
                    
            plt.xlabel('Time Index')
            plt.ylabel('Order Parameter r(t)')
            plt.title('Synchronization Dynamics Across Simulations')
            plt.grid(True, alpha=0.3)
            plt.show()
            
        # If we have steady state synchronization vs coupling strength,
        # plot the critical coupling transition
        if 'steady_state_sync' in feature_data and 'coupling_strength' in feature_data:
            steady_state = feature_data['steady_state_sync']['data']['value']
            k_values = feature_data['coupling_strength']['data']['value']
            
            # Sort by coupling strength
            sorted_indices = np.argsort(k_values)
            k_sorted = k_values[sorted_indices]
            r_sorted = steady_state[sorted_indices]
            
            plt.figure(figsize=(8, 5))
            plt.scatter(k_sorted, r_sorted, c=k_sorted, cmap='viridis', s=50, alpha=0.8)
            plt.xlabel('Coupling Strength (K)')
            plt.ylabel('Steady State Order Parameter r')
            plt.title('Phase Transition to Synchronization')
            plt.grid(True, alpha=0.3)
            
            # Estimate critical coupling
            if len(k_sorted) > 5:
                from scipy.interpolate import interp1d
                
                # Smooth the curve
                k_smooth = np.linspace(k_sorted.min(), k_sorted.max(), 100)
                r_smooth = interp1d(k_sorted, r_sorted, kind='cubic')(k_smooth)
                
                # Estimate critical point (steepest slope)
                slopes = np.gradient(r_smooth, k_smooth)
                critical_idx = np.argmax(slopes)
                critical_k = k_smooth[critical_idx]
                
                plt.axvline(x=critical_k, color='red', linestyle='--', 
                          label=f'Critical K ≈ {critical_k:.4f}')
                plt.legend()
            
            plt.show()
    
    if 'network_type' in feature_data and 'adjacency_matrix' in feature_data:
        # Plot example adjacency matrices for different network types
        adj_matrices = feature_data['adjacency_matrix']['data']
        network_types = feature_data['network_type']['data']['value']
        
        # Show one example of each network type
        unique_networks = set()
        for i, network_info in enumerate(network_types):
            network_type = network_info.get('network_type')
            if network_type and network_type not in unique_networks and len(unique_networks) < 3:
                unique_networks.add(network_type)
                
                plt.figure(figsize=(5, 5))
                plt.imshow(adj_matrices[i], cmap='Blues')
                plt.colorbar()
                plt.title(f'Adjacency Matrix: {network_type}')
                plt.show()


def export_for_pytorch(dataset_id, file_path=None):
    """
    Export a dataset in a format suitable for PyTorch.
    
    Parameters:
    -----------
    dataset_id : int
        ID of the dataset to export
    file_path : str, optional
        Directory where the dataset should be saved
        
    Returns:
    --------
    str
        Path to the exported dataset
    """
    # Export the dataset
    export_path = export_ml_dataset(dataset_id, file_path, format='numpy')
    
    # Print a usage example
    print("Dataset exported for PyTorch.")
    print("Example usage:\n")
    print("```python")
    print("import torch")
    print("import numpy as np")
    print("from torch.utils.data import Dataset, DataLoader")
    print("")
    print("class KuramotoDataset(Dataset):")
    print("    def __init__(self, root_dir, split='train'):")
    print("        self.root_dir = f'{root_dir}/{split}'")
    print("        self.input_data = np.load(f'{self.root_dir}/order_parameter_timeseries.npy')")
    print("        self.targets = np.load(f'{self.root_dir}/steady_state_sync.npy')")
    print("")
    print("    def __len__(self):")
    print("        return len(self.targets)")
    print("")
    print("    def __getitem__(self, idx):")
    print("        x = torch.FloatTensor(self.input_data[idx])")
    print("        y = torch.FloatTensor([self.targets[idx]])")
    print("        return x, y")
    print("")
    print(f"# Create data loaders")
    print(f"train_dataset = KuramotoDataset('{export_path}', 'train')")
    print(f"train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)")
    print(f"val_dataset = KuramotoDataset('{export_path}', 'validation')")
    print(f"val_loader = DataLoader(val_dataset, batch_size=32)")
    print("```")
    
    return export_path


if __name__ == "__main__":
    print("Kuramoto Model ML Helper")
    print("Available functions:")
    print("1. create_critical_coupling_dataset - Create dataset for critical coupling analysis")
    print("2. create_network_topology_dataset - Create dataset comparing network topologies")
    print("3. visualize_dataset - Visualize features in a dataset")
    print("4. export_for_pytorch - Export a dataset for use with PyTorch")
    
    # Example usage
    print("\nExample usage for generating a critical coupling dataset:")
    print("dataset_id = create_critical_coupling_dataset()")
    print("visualize_dataset(dataset_id)")