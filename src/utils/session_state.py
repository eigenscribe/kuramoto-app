"""
Manage session state initialization and updates for the Kuramoto Model Simulator.
"""

import streamlit as st
import numpy as np
import json
import pickle

def initialize_session_state():
    """Initialize all session state variables with default values."""
    # Initialize refresh state for network refresh button
    if 'refresh_network' not in st.session_state:
        st.session_state.refresh_network = False
        
    # Initialize session state for configuration loading
    if 'loaded_config' not in st.session_state:
        st.session_state.loaded_config = None
        
    # Initialize session state for JSON example
    if 'json_example' not in st.session_state:
        st.session_state.json_example = ""
    
    # Initialize session state for parameters
    if 'n_oscillators' not in st.session_state:
        st.session_state.n_oscillators = 10
    if 'coupling_strength' not in st.session_state:
        st.session_state.coupling_strength = 1.0
    if 'freq_type' not in st.session_state:
        st.session_state.freq_type = "Normal"
    if 'freq_mean' not in st.session_state:
        st.session_state.freq_mean = 0.0
    if 'freq_std' not in st.session_state:
        st.session_state.freq_std = 1.0
    if 'freq_min' not in st.session_state:
        st.session_state.freq_min = -1.0
    if 'freq_max' not in st.session_state:
        st.session_state.freq_max = 1.0
    if 'peak1' not in st.session_state:
        st.session_state.peak1 = -1.0
    if 'peak2' not in st.session_state:
        st.session_state.peak2 = 1.0
    if 'custom_freqs' not in st.session_state:
        st.session_state.custom_freqs = "0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -1.5, -2.0"
    if 'simulation_time' not in st.session_state:
        st.session_state.simulation_time = 100.0
    if 'adj_matrix_input' not in st.session_state:
        # Create a default example matrix for a 5x5 ring topology
        default_matrix = "0, 1, 0, 0, 1\n1, 0, 1, 0, 0\n0, 1, 0, 1, 0\n0, 0, 1, 0, 1\n1, 0, 0, 1, 0"
        st.session_state.adj_matrix_input = default_matrix
        
    # For auto-adjusting oscillator count based on matrix dimensions
    if 'next_n_oscillators' not in st.session_state:
        st.session_state.next_n_oscillators = None

def process_loaded_config():
    """Apply loaded configuration from database if available."""
    if st.session_state.loaded_config is not None:
        config = st.session_state.loaded_config
        
        # Update session state with configuration values
        st.session_state.n_oscillators = config['n_oscillators']
        st.session_state.coupling_strength = config['coupling_strength']
        st.session_state.simulation_time = config['simulation_time']
        st.session_state.random_seed = int(config['random_seed'])
        st.session_state.network_type = config['network_type']
        st.session_state.freq_type = config['frequency_distribution']
        
        # Update frequency distribution parameters based on type
        freq_params = config.get('frequency_params', {})
        if freq_params:
            try:
                freq_params = json.loads(freq_params)
            except:
                # It's already a dictionary
                pass
                
            if config['frequency_distribution'] == "Normal":
                st.session_state.freq_mean = freq_params.get('mean', 0.0)
                st.session_state.freq_std = freq_params.get('std', 1.0)
            elif config['frequency_distribution'] == "Uniform":
                st.session_state.freq_min = freq_params.get('min', -1.0)
                st.session_state.freq_max = freq_params.get('max', 1.0)
            elif config['frequency_distribution'] == "Bimodal":
                st.session_state.peak1 = freq_params.get('peak1', -1.0)
                st.session_state.peak2 = freq_params.get('peak2', 1.0)
            elif config['frequency_distribution'] == "Custom":
                if 'custom_values' in freq_params and isinstance(freq_params['custom_values'], list):
                    st.session_state.custom_freqs = ', '.join(map(str, freq_params['custom_values']))
                elif 'values' in freq_params and isinstance(freq_params['values'], list):
                    st.session_state.custom_freqs = ', '.join(map(str, freq_params['values']))
        
        # Handle custom adjacency matrix if present
        if config['network_type'] == "Custom Adjacency Matrix" and config.get('adjacency_matrix') is not None:
            process_adjacency_matrix(config['adjacency_matrix'])
        
        # Clear the loaded config to prevent reapplying it on next rerun
        st.session_state.loaded_config = None

def process_adjacency_matrix(matrix_data):
    """Process and store adjacency matrix in session state."""
    try:
        matrix = matrix_data
        if isinstance(matrix, bytes):
            matrix = pickle.loads(matrix)
        elif isinstance(matrix, list):
            # Convert list to numpy array if it's still a list
            matrix = np.array(matrix)
        
        # Make sure no self-loops (diagonal elements should be zero)
        if hasattr(matrix, 'shape') and matrix.shape[0] == matrix.shape[1]:
            np.fill_diagonal(matrix, 0)
            
            # Convert matrix to string representation for the text area
            matrix_str = ""
            for row in matrix:
                matrix_str += ", ".join(str(val) for val in row) + "\n"
            st.session_state.adj_matrix_input = matrix_str.strip()
            
            # Store the matrix for later use in this session
            st.session_state.loaded_adj_matrix = matrix
            
    except Exception as e:
        st.warning(f"Could not load custom adjacency matrix: {str(e)}")

def process_oscillator_count_update():
    """Update oscillator count if matrix dimensions changed."""
    if st.session_state.next_n_oscillators is not None:
        st.session_state.n_oscillators = st.session_state.next_n_oscillators
        st.session_state.next_n_oscillators = None  # Clear the pending update