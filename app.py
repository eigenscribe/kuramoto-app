"""
Kuramoto Model Simulator - Main Entry Point

This file serves as the entry point for the Streamlit application,
implementing a model of coupled oscillators with interactive visualizations.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime

# Import modules from our package
from src.models.kuramoto import KuramotoModel
from src.components.sidebar import render_sidebar
from src.components.network import render_network_tab
from src.components.frequencies import render_frequencies_tab, generate_frequencies
from src.components.animation import render_animation_tab
from src.components.database_operations import render_database_tab
from src.components.visualization import (
    plot_oscillator_phases, plot_order_parameter, 
    plot_phase_evolution, plot_frequency_histogram
)
from src.database.db import store_simulation

# Set up page configuration
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🌀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
/* Main styling for the entire application */
body {
    font-family: 'Aclonica', sans-serif;
    margin: 0;
    padding: 0;
    color: white;
}

/* Adjust Streamlit default elements */
.stApp {
    background-color: #121212;
}

/* Add more CSS as needed */
</style>
""", unsafe_allow_html=True)

# Load CSS file if it exists
css_path = "src/static/css/styles.css"
if os.path.exists(css_path):
    with open(css_path, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# App title
st.title("🌀 Kuramoto Model Simulator")
st.markdown(
    """
    Explore phase synchronization dynamics in networks of coupled oscillators.
    Configure the network structure, frequency distribution, and coupling parameters
    to see how synchronization emerges.
    """
)

# Session state to store simulation results
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'times' not in st.session_state:
    st.session_state.times = None
if 'phases' not in st.session_state:
    st.session_state.phases = None
if 'order_parameter' not in st.session_state:
    st.session_state.order_parameter = None
if 'params' not in st.session_state:
    st.session_state.params = None
if 'adjacency_matrix' not in st.session_state:
    st.session_state.adjacency_matrix = None
if 'frequencies' not in st.session_state:
    st.session_state.frequencies = None

# Main tabs
main_tabs = st.tabs([
    "🔬 Simulation", 
    "🔗 Network", 
    "🌊 Frequencies", 
    "🎬 Animation", 
    "📊 Database"
])

# Render the sidebar and get simulation parameters
params, submit_button = render_sidebar(st.session_state.get('config', None))

# If a configuration is loaded from the database, update parameters
if 'load_config' in st.session_state and st.session_state.load_config:
    config = st.session_state.load_config
    # Clear the flag to prevent reloading
    st.session_state.load_config = None
    st.experimental_rerun()

# Helper function to run a simulation
def run_simulation(model, params, adjacency_matrix=None, frequencies=None):
    """
    Run a Kuramoto model simulation with the specified parameters.
    
    Parameters:
    -----------
    model : KuramotoModel class
        The Kuramoto model class to instantiate
    params : dict
        Simulation parameters
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix for network connectivity
    frequencies : ndarray, optional
        Custom natural frequencies for oscillators
        
    Returns:
    --------
    model : KuramotoModel
        The instantiated model object
    times : ndarray
        Time points of the simulation
    phases : ndarray
        Phases of oscillators at each time point
    order_parameter : ndarray
        Order parameter values over time
    """
    # Extract parameters
    n_oscillators = params.get('n_oscillators', 10)
    coupling_strength = params.get('coupling_strength', 1.0)
    simulation_time = params.get('simulation_time', 20.0)
    time_step = params.get('time_step', 0.05)
    random_seed = params.get('random_seed', 42)
    
    # Instantiate the model
    kuramoto_model = model(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed,
        adjacency_matrix=adjacency_matrix
    )
    
    # Run the simulation
    times, phases, order_parameter = kuramoto_model.simulate()
    
    return kuramoto_model, times, phases, order_parameter

# Helper function to get frequency parameters
def get_frequency_params(params):
    """
    Extract frequency distribution parameters from the simulation parameters.
    
    Parameters:
    -----------
    params : dict
        Simulation parameters
        
    Returns:
    --------
    dict
        Frequency distribution parameters
    str
        Frequency distribution type
    """
    freq_distribution = params.get('frequency_distribution', 'Normal')
    freq_params = params.get('freq_params', {})
    
    return freq_params, freq_distribution

# Simulation tab
with main_tabs[0]:
    st.markdown("## Kuramoto Model Simulation")
    
    # Information about the model
    with st.expander("About the Kuramoto Model", expanded=False):
        st.markdown("""
        The Kuramoto model describes the phase dynamics of a system of coupled oscillators.
        Each oscillator has its own natural frequency and is coupled to other oscillators
        according to a specified network structure.
        
        The governing differential equation for each oscillator is:
        
        $$\\frac{d\\theta_i}{dt} = \\omega_i + \\frac{K}{N} \\sum_{j=1}^{N} A_{ij} \\sin(\\theta_j - \\theta_i)$$
        
        where:
        - $\\theta_i$ is the phase of oscillator $i$
        - $\\omega_i$ is the natural frequency of oscillator $i$
        - $K$ is the coupling strength
        - $N$ is the number of oscillators
        - $A_{ij}$ is the adjacency matrix defining the network structure
        
        The degree of synchronization is measured by the order parameter $r(t)$:
        
        $$r(t) = \\left| \\frac{1}{N} \\sum_{j=1}^{N} e^{i\\theta_j(t)} \\right|$$
        
        which ranges from 0 (no synchronization) to 1 (perfect synchronization).
        """)
    
    # Run simulation or display results
    if submit_button or st.session_state.simulation_run:
        if submit_button:
            # Process parameters from sidebar
            st.session_state.params = params
            
            # Get adjacency matrix from network tab if available
            adjacency_matrix = st.session_state.get('adjacency_matrix', None)
            
            # Get custom frequencies if available
            frequencies = st.session_state.get('frequencies', None)
            
            # Generate frequencies if not provided
            if frequencies is None:
                freq_params, freq_distribution = get_frequency_params(params)
                frequencies = generate_frequencies(
                    freq_distribution, 
                    params['n_oscillators'], 
                    freq_params, 
                    params['random_seed']
                )
            
            # Run the simulation
            with st.spinner("Running simulation..."):
                model, times, phases, order_parameter = run_simulation(
                    KuramotoModel, 
                    params, 
                    adjacency_matrix=adjacency_matrix, 
                    frequencies=frequencies
                )
                
                # Store results in session state
                st.session_state.model = model
                st.session_state.times = times
                st.session_state.phases = phases
                st.session_state.order_parameter = order_parameter
                st.session_state.frequencies = model.frequencies
                st.session_state.adjacency_matrix = model.adjacency_matrix
                st.session_state.simulation_run = True
            
            # Success message
            st.success("Simulation completed successfully!")
        
        # Display simulation results
        if st.session_state.simulation_run:
            # Get results from session state
            model = st.session_state.model
            times = st.session_state.times
            phases = st.session_state.phases
            order_parameter = st.session_state.order_parameter
            
            # Create two columns for visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Initial Oscillator Positions")
                fig = plot_oscillator_phases(phases, times, time_idx=0)
                st.pyplot(fig)
                
                st.subheader("Order Parameter Evolution")
                fig = plot_order_parameter(times, order_parameter)
                st.pyplot(fig)
            
            with col2:
                st.subheader("Final Oscillator Positions")
                fig = plot_oscillator_phases(phases, times, time_idx=-1)
                st.pyplot(fig)
                
                st.subheader("Phase Evolution")
                fig = plot_phase_evolution(phases, times)
                st.pyplot(fig)
            
            # Display frequency distribution
            st.subheader("Frequency Distribution")
            freq_params, freq_distribution = get_frequency_params(st.session_state.params)
            fig = plot_frequency_histogram(model.frequencies, freq_distribution, freq_params)
            st.pyplot(fig)
            
            # Save to database option
            st.markdown("### Save Simulation Results")
            
            if st.button("Save to Database"):
                with st.spinner("Saving to database..."):
                    # Get frequency distribution info
                    freq_type = st.session_state.params.get('frequency_distribution', 'Normal')
                    freq_params = st.session_state.params.get('freq_params', {})
                    
                    # Store in database
                    sim_id = store_simulation(
                        model, 
                        times, 
                        phases, 
                        order_parameter, 
                        model.frequencies, 
                        freq_type, 
                        freq_params, 
                        model.adjacency_matrix
                    )
                    
                    if sim_id:
                        st.success(f"Simulation saved to database with ID: {sim_id}")
                    else:
                        st.error("Failed to save simulation to database.")
    else:
        st.info("Configure parameters in the sidebar and click 'Run Simulation' to start.")
        
        # Show some example images
        st.markdown("### Sample Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            try:
                st.image("src/static/images/1p.png", caption="Sample Oscillator Visualization", use_container_width=True)
            except:
                st.warning("Sample image not found.")
        
        with col2:
            st.markdown("""
            The Kuramoto model displays different synchronization behaviors depending on:
            
            - **Coupling Strength**: Stronger coupling leads to more synchronization
            - **Frequency Distribution**: Narrow distributions synchronize more easily
            - **Network Structure**: Certain network topologies enhance synchronization
            
            Run a simulation to explore these dynamics interactively!
            """)

# Network tab
with main_tabs[1]:
    # Get network parameters from the sidebar
    network_params = {
        'n_oscillators': params.get('n_oscillators', 10),
        'network_type': params.get('network_type', 'All-to-All')
    }
    
    # Add network-specific parameters if available
    for param in ['connection_prob', 'k_neighbors', 'rewire_prob', 'sf_m', 'dimensions']:
        if param in params:
            network_params[param] = params[param]
    
    # Render the network tab
    adjacency_matrix = render_network_tab(network_params)
    
    # Store adjacency matrix in session state if returned
    if adjacency_matrix is not None:
        st.session_state.adjacency_matrix = adjacency_matrix

# Frequencies tab
with main_tabs[2]:
    # Get frequency parameters from the sidebar
    frequency_params = {
        'n_oscillators': params.get('n_oscillators', 10),
        'frequency_distribution': params.get('frequency_distribution', 'Normal'),
        'random_seed': params.get('random_seed', 42),
        'freq_params': params.get('freq_params', {})
    }
    
    # Render the frequencies tab
    frequencies = render_frequencies_tab(frequency_params)
    
    # Store frequencies in session state if returned
    if frequencies is not None:
        st.session_state.frequencies = frequencies

# Animation tab
with main_tabs[3]:
    if st.session_state.simulation_run:
        # Render the animation tab
        render_animation_tab(
            st.session_state.model,
            st.session_state.times,
            st.session_state.phases,
            st.session_state.order_parameter
        )
    else:
        st.warning("Please run a simulation first to enable animations.")

# Database tab
with main_tabs[4]:
    # Render the database tab
    selected_config = render_database_tab()
    
    # If a configuration is selected, store it in session state for loading
    if selected_config:
        st.session_state.load_config = selected_config
        # Force rerun to apply the configuration
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #888;'>
    Kuramoto Model Simulator | Created with Streamlit | 
    <a href='https://en.wikipedia.org/wiki/Kuramoto_model' target='_blank'>Learn more about the Kuramoto model</a>
    </div>
    """,
    unsafe_allow_html=True
)