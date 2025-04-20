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

# Try to load style.css file directly
try:
    with open("styles.css", "r") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
except:
    # Fallback if the styles.css file can't be read
    # Load the image file for background
    try:
        with open("src/static/images/wisp.jpg", "rb") as f:
            img_data = f.read()
            import base64
            img_base64 = base64.b64encode(img_data).decode()
            background_image = f"data:image/jpeg;base64,{img_base64}"
    except:
        # Fallback if image is not found
        background_image = None

    # Comprehensive CSS with inline background image
    st.markdown(f"""
    <style>
    /* Main styling for the entire application */
    body {{
        font-family: 'Aclonica', sans-serif;
        margin: 0;
        padding: 0;
        color: white;
    }}

    /* Adjust Streamlit default elements */
    .stApp {{
        background-color: #121212;
        background-image: {f"url('{background_image}')" if background_image else "none"};
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    /* Gradient text styles */
    .gradient_text1 {{
        font-weight: bold !important;
        background: linear-gradient(to right bottom, #00e8ff, #14a5ff, #8138ff) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
    }}

    .gradient_text2 {{
        font-weight: bold !important;
        background: linear-gradient(to right bottom, #ff8a00, #ff5e62, #ff2f92) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
    }}

    /* Custom animation for oscillator visualization */
    @keyframes pulse {{
        0% {{
            transform: scale(1);
            opacity: 0.8;
        }}
        50% {{
            transform: scale(1.1);
            opacity: 1;
        }}
        100% {{
            transform: scale(1);
            opacity: 0.8;
        }}
    }}

    /* Glow effects for various elements */
    .glow {{
        box-shadow: 0 0 10px 2px rgba(0, 232, 255, 0.7),
                    0 0 20px 4px rgba(20, 165, 255, 0.5),
                    0 0 30px 6px rgba(129, 56, 255, 0.3);
        border-radius: 10px;
    }}

    /* Custom tabs styling */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 5px;
    }}

    .stTabs [data-baseweb="tab"] {{
        background-color: rgba(30, 30, 30, 0.6) !important;
        border-radius: 15px 15px 0px 0px !important;
        padding: 5px 20px !important;
        color: white !important;
        transition: all 0.3s ease !important;
    }}

    .stTabs [data-baseweb="tab-highlight"] {{
        background: linear-gradient(to right, #00e8ff, #14a5ff, #8138ff) !important;
        border-radius: 15px 15px 0px 0px !important;
        height: 4px !important;
    }}

    .stTabs [data-baseweb="tab-panel"] {{
        background-color: rgba(30, 30, 30, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        -webkit-backdrop-filter: blur(10px) !important;
        border-radius: 0px 15px 15px 15px !important;
        padding: 20px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.3) !important;
    }}

    /* Sidebar styling */
    [data-testid="stSidebar"] {{
        background-color: rgba(20, 20, 20, 0.8) !important;
        backdrop-filter: blur(15px) !important;
        -webkit-backdrop-filter: blur(15px) !important;
        border-right: 1px solid rgba(80, 80, 80, 0.2) !important;
    }}

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {{
        background: linear-gradient(to right, #00e8ff, #14a5ff, #8138ff) !important;
        -webkit-background-clip: text !important;
        background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        color: transparent !important;
        font-weight: bold !important;
    }}

    /* Custom button styles */
    div.stButton > button:first-child {{
        background: linear-gradient(45deg, #00e8ff, #14a5ff, #8138ff) !important;
        color: white !important;
        font-weight: bold !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 10px 24px !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 0 15px rgba(20, 165, 255, 0.5) !important;
    }}

    div.stButton > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 20px rgba(20, 165, 255, 0.7) !important;
    }}

    /* Slider styling */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {{
        background: linear-gradient(to bottom, #00e8ff, #14a5ff, #8138ff) !important;
        box-shadow: 0 0 10px rgba(20, 165, 255, 0.7) !important;
    }}

    [data-testid="stSlider"] [data-baseweb="slider"] div[role="progressbar"] {{
        background: linear-gradient(to right, #00e8ff, #8138ff) !important;
    }}

    /* Make elements more rounded */
    [data-testid="stContainer"],
    [data-testid="stMarkdown"],
    [data-testid="stImage"],
    [data-testid="stText"],
    [data-testid="stButton"],
    [data-testid="stFileUploader"],
    [data-testid="stSelectbox"],
    [data-testid="stTextInput"],
    [data-testid="stNumberInput"],
    [data-testid="stSlider"],
    [data-testid="stCheckbox"] {{
        border-radius: 15px !important;
    }}

    /* Footer styling */
    footer {{
        visibility: hidden;
    }}
    </style>
    """, unsafe_allow_html=True)

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