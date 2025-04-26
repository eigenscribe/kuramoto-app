"""
Utilities for handling JSON configurations in the Kuramoto Model Simulator.
"""

import json
import numpy as np
import streamlit as st

def parse_json_parameters(json_string):
    """
    Parse a JSON string containing Kuramoto simulation parameters.
    
    Expected format:
    {
        "n_oscillators": 10,
        "coupling_strength": 1.0,
        "network_type": "All-to-All", 
        "simulation_time": 10.0,
        "time_step": 0.1,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.2
        },
        "adjacency_matrix": [[0,1,1,...], [1,0,1,...], ...] (optional)
    }
    
    Returns:
    dict: Parameters dictionary containing all parsed values, or error message
    """
    try:
        # Parse the JSON string
        params = json.loads(json_string)
        
        # Initialize the result dictionary with default values
        result = {
            "n_oscillators": 10,
            "coupling_strength": 1.0,
            "network_type": "All-to-All",
            "simulation_time": 10.0,
            "time_step": 0.1,
            "random_seed": 42,
            "frequency_distribution": "Normal",
            "frequency_parameters": {
                "mean": 0.0,
                "std": 0.2
            },
            "adjacency_matrix": None
        }
        
        # Update with values from the JSON
        if "n_oscillators" in params:
            result["n_oscillators"] = int(params["n_oscillators"])
            
        if "coupling_strength" in params:
            result["coupling_strength"] = float(params["coupling_strength"])
            
        if "network_type" in params:
            valid_types = ["All-to-All", "Nearest Neighbor", "Random", "Custom Adjacency Matrix"]
            if params["network_type"] in valid_types:
                result["network_type"] = params["network_type"]
            
        if "simulation_time" in params:
            result["simulation_time"] = float(params["simulation_time"])
            
        # time_step is now automatically calculated based on oscillator frequencies
        # We keep this for backward compatibility with existing JSON configs
        if "time_step" in params:
            # We acknowledge the parameter but don't use it directly
            pass
            
        if "random_seed" in params:
            result["random_seed"] = int(params["random_seed"])
            
        if "frequency_distribution" in params:
            valid_distributions = ["Normal", "Uniform", "Custom", "Golden Ratio", "Bimodal"]
            if params["frequency_distribution"] in valid_distributions:
                result["frequency_distribution"] = params["frequency_distribution"]
            
        if "frequency_parameters" in params:
            fp = params["frequency_parameters"]
            if "mean" in fp and result["frequency_distribution"] == "Normal":
                result["frequency_parameters"]["mean"] = float(fp["mean"])
            if "std" in fp and result["frequency_distribution"] == "Normal":
                result["frequency_parameters"]["std"] = float(fp["std"])
            if "min" in fp and result["frequency_distribution"] == "Uniform":
                result["frequency_parameters"]["min"] = float(fp["min"])
            if "max" in fp and result["frequency_distribution"] == "Uniform":
                result["frequency_parameters"]["max"] = float(fp["max"])
            if "custom_values" in fp and result["frequency_distribution"] == "Custom":
                # Convert custom frequency values to floats
                result["frequency_parameters"]["custom_values"] = [float(x) for x in fp["custom_values"]]
        
        # Process the adjacency matrix if provided
        if "adjacency_matrix" in params:
            matrix_data = params["adjacency_matrix"]
            if isinstance(matrix_data, list) and len(matrix_data) > 0:
                try:
                    # Convert to numpy array and validate
                    adj_matrix = np.array(matrix_data, dtype=float)
                    
                    # Check if the matrix is square
                    if adj_matrix.shape[0] == adj_matrix.shape[1]:
                        result["adjacency_matrix"] = adj_matrix
                        # If adjacency matrix is provided, force network type to Custom
                        result["network_type"] = "Custom Adjacency Matrix"
                    else:
                        return None, f"Adjacency matrix must be square. Current shape: {adj_matrix.shape}"
                except Exception as e:
                    return None, f"Error processing adjacency matrix: {str(e)}"
        
        return result, None
        
    except Exception as e:
        # Return error message if parsing fails
        return None, str(e)

def update_session_from_json(params):
    """
    Update session state with parameters from parsed JSON.
    
    Args:
        params (dict): Dictionary of parsed parameters
    """
    # Update session state with the parsed parameters
    st.session_state.n_oscillators = params["n_oscillators"]
    st.session_state.coupling_strength = params["coupling_strength"]
    st.session_state.network_type = params["network_type"]
    st.session_state.simulation_time = params["simulation_time"]
    # time_step is no longer used, it's automatically calculated
    st.session_state.random_seed = params["random_seed"]
    st.session_state.freq_type = params["frequency_distribution"]
    
    # Update frequency parameters based on distribution type
    if params["frequency_distribution"] == "Normal":
        st.session_state.freq_mean = params["frequency_parameters"]["mean"]
        st.session_state.freq_std = params["frequency_parameters"]["std"]
    elif params["frequency_distribution"] == "Uniform":
        st.session_state.freq_min = params["frequency_parameters"]["min"]
        st.session_state.freq_max = params["frequency_parameters"]["max"]
    elif params["frequency_distribution"] == "Custom" and "custom_values" in params["frequency_parameters"]:
        st.session_state.custom_freqs = ", ".join(str(x) for x in params["frequency_parameters"]["custom_values"])
    
    # Handle custom adjacency matrix if present
    if params["adjacency_matrix"] is not None:
        matrix = params["adjacency_matrix"]
        
        # Convert matrix to string representation for the text area
        matrix_str = ""
        for row in matrix:
            matrix_str += ", ".join(str(val) for val in row) + "\n"
        
        # Update session state for adjacency matrix
        st.session_state.adj_matrix_input = matrix_str.strip()
        st.session_state.loaded_adj_matrix = matrix

def generate_example_json():
    """
    Generate example JSON configuration.
    
    Returns:
        dict: Example configuration
    """
    return {
        "n_oscillators": 10,
        "coupling_strength": 1.0,
        "network_type": "All-to-All", 
        "simulation_time": 100.0,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.2
        }
    }

def generate_small_world_example():
    """
    Generate small-world network example.
    
    Returns:
        dict: Small-world network example
    """
    # Generate a sample small-world network
    n = 10
    sample_matrix = np.zeros((n, n))
    for i in range(n):
        # Connect to neighbors
        for j in range(1, 3):
            sample_matrix[i, (i+j) % n] = 1
            sample_matrix[i, (i-j) % n] = 1
            
    # Add a few random long-range connections
    np.random.seed(42)
    for _ in range(5):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j and sample_matrix[i, j] == 0:
            sample_matrix[i, j] = 1
            sample_matrix[j, i] = 1
            
    # Create example with matrix
    return {
        "n_oscillators": n,
        "coupling_strength": 0.8,
        "network_type": "Custom Adjacency Matrix",
        "simulation_time": 100.0,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.1
        },
        "adjacency_matrix": sample_matrix.tolist()
    }