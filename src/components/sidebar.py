"""
Sidebar components for the Kuramoto Model Simulator.
Contains all sidebar controls and configuration options.
"""

import streamlit as st
import numpy as np
import json
from src.database.database import save_configuration, get_configuration, list_configurations
from src.utils.simulation import parse_json_parameters, generate_frequencies, create_adjacency_matrix


def render_sidebar():
    """Render the complete sidebar with all controls."""
    st.sidebar.title("🔄 Kuramoto Model")
    st.sidebar.markdown("---")
    
    # Basic parameters
    basic_params = render_basic_parameters()
    
    # Network configuration
    network_params = render_network_configuration(basic_params['n_oscillators'])
    
    # Frequency distribution
    freq_params = render_frequency_distribution(basic_params['n_oscillators'])
    
    # Time controls
    time_params = render_time_controls()
    
    # Configuration management
    render_configuration_management()
    
    # Examples
    render_examples()
    
    # Combine all parameters
    all_params = {**basic_params, **network_params, **freq_params, **time_params}
    
    return all_params


def render_basic_parameters():
    """Render basic simulation parameters."""
    st.sidebar.subheader("Basic Parameters")
    
    n_oscillators = st.sidebar.slider(
        "Number of Oscillators", 
        min_value=3, 
        max_value=100, 
        value=10,
        help="Total number of oscillators in the system"
    )
    
    coupling_strength = st.sidebar.slider(
        "Coupling Strength (K)", 
        min_value=0.0, 
        max_value=10.0, 
        value=1.0, 
        step=0.1,
        help="Strength of interaction between oscillators"
    )
    
    return {
        'n_oscillators': n_oscillators,
        'coupling_strength': coupling_strength
    }


def render_network_configuration(n_oscillators):
    """Render network configuration options."""
    st.sidebar.subheader("Network Configuration")
    
    network_type = st.sidebar.selectbox(
        "Network Type",
        ["Random", "All-to-All", "Small World", "Scale-Free"],
        index=0,
        help="Type of network connectivity between oscillators"
    )
    
    adjacency_matrix = None
    network_stats = None
    
    # Network-specific parameters
    if network_type == "Random":
        connection_prob = st.sidebar.slider(
            "Connection Probability", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.3, 
            step=0.05,
            help="Probability of connection between any two oscillators"
        )
        adjacency_matrix = create_adjacency_matrix(
            n_oscillators, network_type, connection_probability=connection_prob
        )
        
    elif network_type == "Small World":
        k = st.sidebar.slider(
            "Nearest Neighbors (k)", 
            min_value=2, 
            max_value=min(10, n_oscillators-1), 
            value=4,
            help="Number of nearest neighbors in ring lattice"
        )
        p = st.sidebar.slider(
            "Rewiring Probability", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.1, 
            step=0.05,
            help="Probability of rewiring each edge"
        )
        adjacency_matrix = create_adjacency_matrix(
            n_oscillators, network_type, k=k, p=p
        )
        
    elif network_type == "Scale-Free":
        m = st.sidebar.slider(
            "Attachment Edges (m)", 
            min_value=1, 
            max_value=min(5, n_oscillators-1), 
            value=2,
            help="Number of edges to attach from each new node"
        )
        adjacency_matrix = create_adjacency_matrix(
            n_oscillators, network_type, m=m
        )
        
    else:  # All-to-All
        adjacency_matrix = create_adjacency_matrix(n_oscillators, network_type)
    
    # Refresh network button
    if st.sidebar.button("🔄 Refresh Network", help="Generate a new random network"):
        st.session_state.refresh_network = True
        st.rerun()
    
    return {
        'network_type': network_type,
        'adjacency_matrix': adjacency_matrix
    }


def render_frequency_distribution(n_oscillators):
    """Render frequency distribution configuration."""
    st.sidebar.subheader("Frequency Distribution")
    
    frequency_distribution = st.sidebar.selectbox(
        "Distribution Type",
        ["Normal", "Uniform", "Bimodal", "Custom"],
        help="Type of natural frequency distribution"
    )
    
    freq_params_dict = {}
    
    if frequency_distribution == "Normal":
        mean = st.sidebar.slider("Mean", -2.0, 2.0, 0.0, 0.1)
        std = st.sidebar.slider("Standard Deviation", 0.1, 2.0, 1.0, 0.1)
        freq_params_dict = {'mean': mean, 'std': std}
        
    elif frequency_distribution == "Uniform":
        low = st.sidebar.slider("Lower Bound", -3.0, 0.0, -1.0, 0.1)
        high = st.sidebar.slider("Upper Bound", 0.0, 3.0, 1.0, 0.1)
        freq_params_dict = {'low': low, 'high': high}
        
    elif frequency_distribution == "Bimodal":
        col1, col2 = st.sidebar.columns(2)
        with col1:
            mean1 = st.slider("Mean 1", -2.0, 0.0, -1.0, 0.1)
            std1 = st.slider("Std 1", 0.1, 1.0, 0.2, 0.05)
        with col2:
            mean2 = st.slider("Mean 2", 0.0, 2.0, 1.0, 0.1)
            std2 = st.slider("Std 2", 0.1, 1.0, 0.2, 0.05)
        weight = st.sidebar.slider("Group 1 Weight", 0.1, 0.9, 0.5, 0.05)
        freq_params_dict = {
            'mean1': mean1, 'mean2': mean2,
            'std1': std1, 'std2': std2,
            'weight': weight
        }
        
    elif frequency_distribution == "Custom":
        freq_params_json = st.sidebar.text_area(
            "Custom Parameters (JSON)",
            '{"mean": 0.0, "std": 1.0}',
            help="Enter custom parameters as JSON"
        )
        parsed_params = parse_json_parameters(freq_params_json)
        if parsed_params:
            freq_params_dict = parsed_params
        else:
            st.sidebar.error("Invalid JSON format")
            freq_params_dict = {'mean': 0.0, 'std': 1.0}
    
    # Generate frequencies
    frequencies, final_freq_params = generate_frequencies(
        n_oscillators, frequency_distribution, freq_params_dict
    )
    
    return {
        'frequency_distribution': frequency_distribution,
        'frequencies': frequencies,
        'freq_params': final_freq_params
    }


def render_time_controls():
    """Render time-related controls."""
    st.sidebar.subheader("Time Controls")
    
    simulation_time = st.sidebar.slider(
        "Simulation Time", 
        min_value=10.0, 
        max_value=200.0, 
        value=100.0, 
        step=10.0,
        help="Total time to simulate the system"
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed", 
        min_value=1, 
        max_value=9999, 
        value=42,
        help="Seed for reproducible random number generation"
    )
    
    return {
        'simulation_time': simulation_time,
        'random_seed': random_seed
    }


def render_configuration_management():
    """Render configuration save/load functionality."""
    with st.sidebar.expander("Save as Preset"):
        preset_name = st.text_input(
            "Preset Name", 
            key="preset_name", 
            placeholder="Enter a name for this configuration"
        )
        
        if st.button("💾 Save Preset", key="save_preset_btn"):
            if preset_name:
                try:
                    # Get current parameters from session state or defaults
                    current_params = st.session_state.get('current_params', {})
                    
                    config_id = save_configuration(
                        name=preset_name,
                        n_oscillators=current_params.get('n_oscillators', 10),
                        coupling_strength=current_params.get('coupling_strength', 1.0),
                        simulation_time=current_params.get('simulation_time', 100.0),
                        time_step=0.01,  # Default value
                        random_seed=current_params.get('random_seed', 42),
                        network_type=current_params.get('network_type', 'Random'),
                        frequency_distribution=current_params.get('frequency_distribution', 'Normal'),
                        frequency_params=current_params.get('freq_params', {}),
                        adjacency_matrix=current_params.get('adjacency_matrix', None)
                    )
                    st.success(f"✅ Saved preset '{preset_name}' (ID: {config_id})")
                except Exception as e:
                    st.error(f"❌ Error saving preset: {str(e)}")
            else:
                st.warning("⚠️ Please enter a preset name")


def render_examples():
    """Render example configurations."""
    with st.sidebar.expander("Examples", expanded=False):
        st.markdown("**Quick Start Examples:**")
        
        if st.button("🎯 Synchronization Demo", key="sync_demo"):
            st.session_state.update({
                'n_oscillators': 20,
                'coupling_strength': 2.0,
                'frequency_distribution': 'Normal',
                'network_type': 'All-to-All',
                'simulation_time': 50.0
            })
            st.rerun()
        
        if st.button("🌐 Small World Network", key="small_world"):
            st.session_state.update({
                'n_oscillators': 30,
                'coupling_strength': 1.5,
                'frequency_distribution': 'Uniform',
                'network_type': 'Small World',
                'simulation_time': 100.0
            })
            st.rerun()
        
        if st.button("⚡ Bimodal Frequencies", key="bimodal"):
            st.session_state.update({
                'n_oscillators': 25,
                'coupling_strength': 1.0,
                'frequency_distribution': 'Bimodal',
                'network_type': 'Random',
                'simulation_time': 80.0
            })
            st.rerun()
        
        if st.button("🕸️ Scale-Free Network", key="scale_free"):
            st.session_state.update({
                'n_oscillators': 40,
                'coupling_strength': 0.8,
                'frequency_distribution': 'Normal',
                'network_type': 'Scale-Free',
                'simulation_time': 120.0
            })
            st.rerun()