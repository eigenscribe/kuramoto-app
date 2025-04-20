"""
Sidebar component for the Kuramoto simulator.
This module contains functions to render the sidebar with simulation parameters.
"""

import streamlit as st
import numpy as np
import json

def render_sidebar(config=None):
    """
    Render the sidebar with simulation parameters.
    
    Parameters:
    -----------
    config : dict, optional
        Configuration to pre-populate the form
        
    Returns:
    --------
    dict
        Dictionary of simulation parameters
    """
    st.sidebar.markdown("# 🌀 Kuramoto Model Simulator")
    st.sidebar.markdown("### Configure Simulation Parameters")
    st.sidebar.markdown("---")
    
    # Default values
    default_values = {
        'n_oscillators': 10,
        'coupling_strength': 1.0,
        'simulation_time': 20.0,
        'time_step': 0.05,
        'random_seed': 42,
        'network_type': 'All-to-All',
        'frequency_distribution': 'Normal',
        'freq_params': json.dumps({'mean': 0.0, 'std': 1.0})
    }
    
    # Use configuration values if provided
    if config:
        for key in default_values:
            if key in config:
                default_values[key] = config[key]
        
        if 'frequency_params' in config and config['frequency_params']:
            default_values['freq_params'] = json.dumps(config['frequency_params'])
    
    # Create form for parameters
    with st.sidebar.form(key='simulation_params'):
        # Network Parameters (Blue Group)
        st.markdown("#### Network Parameters")
        n_oscillators = st.slider("Number of Oscillators", 2, 100, default_values['n_oscillators'])
        
        network_types = ["All-to-All", "Random", "Small-World", "Scale-Free", "Lattice", "Custom"]
        network_type = st.selectbox(
            "Network Type", 
            options=network_types,
            index=network_types.index(default_values['network_type']) if default_values['network_type'] in network_types else 0
        )
        
        # Only show these parameters for specific network types
        if network_type == "Random":
            connection_prob = st.slider("Connection Probability", 0.0, 1.0, 0.3, step=0.05)
        elif network_type == "Small-World":
            k_neighbors = st.slider("K Nearest Neighbors", 2, min(10, n_oscillators//2*2), 4, step=2)
            rewire_prob = st.slider("Rewiring Probability", 0.0, 1.0, 0.1, step=0.05)
        elif network_type == "Scale-Free":
            sf_m = st.slider("New Edges (m)", 1, 5, 2)
        elif network_type == "Lattice":
            dimensions = st.slider("Dimensions", 1, 2, 2)
        elif network_type == "Custom":
            st.markdown("Upload a custom adjacency matrix in the Network tab")
        
        # Time Parameters (Green Group)
        st.markdown("#### Time Parameters")
        simulation_time = st.slider("Simulation Time (s)", 1.0, 100.0, default_values['simulation_time'])
        time_step = st.slider("Time Step", 0.01, 0.2, default_values['time_step'], step=0.01)
        
        # Oscillator Parameters (Orange Group)
        st.markdown("#### Oscillator Parameters")
        coupling_strength = st.slider("Coupling Strength (K)", 0.0, 5.0, default_values['coupling_strength'], step=0.1)
        
        freq_distributions = ["Normal", "Uniform", "Bimodal", "Custom"]
        freq_distribution = st.selectbox(
            "Frequency Distribution",
            options=freq_distributions,
            index=freq_distributions.index(default_values['frequency_distribution']) if default_values['frequency_distribution'] in freq_distributions else 0
        )
        
        # Parse frequency parameters from JSON
        freq_params = {}
        if default_values['freq_params']:
            try:
                freq_params = json.loads(default_values['freq_params'])
            except json.JSONDecodeError:
                freq_params = {}
        
        # Distribution-specific parameters
        if freq_distribution == "Normal":
            freq_mean = st.slider("Mean Frequency", -2.0, 2.0, freq_params.get('mean', 0.0), step=0.1)
            freq_std = st.slider("Frequency Std Dev", 0.1, 3.0, freq_params.get('std', 1.0), step=0.1)
        elif freq_distribution == "Uniform":
            freq_min = st.slider("Min Frequency", -5.0, 0.0, freq_params.get('min', -1.0), step=0.1)
            freq_max = st.slider("Max Frequency", 0.0, 5.0, freq_params.get('max', 1.0), step=0.1)
        elif freq_distribution == "Bimodal":
            peak1 = st.slider("Peak 1", -5.0, 0.0, freq_params.get('peak1', -1.0), step=0.1)
            peak2 = st.slider("Peak 2", 0.0, 5.0, freq_params.get('peak2', 1.0), step=0.1)
        elif freq_distribution == "Custom":
            st.markdown("Define custom frequencies in the Frequencies tab")
        
        random_seed = st.number_input("Random Seed", value=default_values['random_seed'], help="Seed for random number generation to ensure reproducibility")
        
        # Form submission button
        submit_button = st.form_submit_button(label="Run Simulation")
    
    # Collect all parameters
    params = {
        'n_oscillators': n_oscillators,
        'coupling_strength': coupling_strength,
        'simulation_time': simulation_time,
        'time_step': time_step,
        'random_seed': random_seed,
        'network_type': network_type,
        'frequency_distribution': freq_distribution
    }
    
    # Add network-specific parameters
    if network_type == "Random":
        params['connection_prob'] = connection_prob
    elif network_type == "Small-World":
        params['k_neighbors'] = k_neighbors
        params['rewire_prob'] = rewire_prob
    elif network_type == "Scale-Free":
        params['sf_m'] = sf_m
    elif network_type == "Lattice":
        params['dimensions'] = dimensions
    
    # Add frequency-specific parameters
    if freq_distribution == "Normal":
        params['freq_params'] = {'mean': freq_mean, 'std': freq_std}
    elif freq_distribution == "Uniform":
        params['freq_params'] = {'min': freq_min, 'max': freq_max}
    elif freq_distribution == "Bimodal":
        params['freq_params'] = {'peak1': peak1, 'peak2': peak2}
    
    return params, submit_button