"""
Kuramoto Model Simulator

A Streamlit-based application for exploring phase synchronization in coupled oscillator networks.
The application is modularized into separate components for maintainability.
"""
import streamlit as st
import numpy as np
import json

# Import UI components
from src.ui.common import (
    setup_page, display_title, display_sidebar_header, 
    initialize_session_state, load_configuration, update_oscillator_count
)
from src.ui.sidebar import (
    oscillator_parameters, frequency_distribution_parameters,
    simulation_time_parameters, random_seed_parameter, network_connectivity_parameters
)
from src.ui.simulation import run_simulation, display_simulation_results
from src.ui.ml_tab import render_ml_tab
from src.ui.database_tab import render_database_tab

# Set up the page
setup_page()

# Initialize session state
initialize_session_state()

# Load configuration if available
load_configuration()

# Update oscillator count if needed
update_oscillator_count()

# Display title
display_title()

# Display sidebar header
display_sidebar_header()

# Sidebar parameters
n_oscillators, coupling_strength = oscillator_parameters()
freq_type, frequencies = frequency_distribution_parameters()
simulation_time, time_step = simulation_time_parameters()
random_seed = random_seed_parameter()
network_type, adj_matrix, connection_probability = network_connectivity_parameters()

# Create tabs for main sections
main_tab, db_tab, ml_tab = st.tabs(["Simulation", "Database", "Machine Learning"])

# Main simulation tab
with main_tab:
    # Add a run button to execute the simulation
    if st.button("Run Simulation", type="primary", key="run_simulation_button"):
        with st.spinner("Running simulation..."):
            # Create a dictionary for auto-optimization and get the flag from session state
            auto_optimize = st.session_state.auto_optimize_on_run if "auto_optimize_on_run" in st.session_state else False
            
            # Run the simulation
            model, times, phases, order_parameter, optimized_time_step = run_simulation(
                n_oscillators=n_oscillators,
                coupling_strength=coupling_strength,
                frequencies=frequencies,
                simulation_time=simulation_time,
                time_step=time_step,
                random_seed=random_seed,
                adjacency_matrix=adj_matrix,
                auto_optimize=auto_optimize
            )
            
            # If the time step was optimized, display a message
            if optimized_time_step is not None:
                st.success(f"""
                Time step was automatically optimized from {time_step:.4f} to {optimized_time_step:.4f}.
                
                This ensures stability and accuracy while maintaining computational efficiency.
                """)
            
            # Store simulation results in session state
            st.session_state.simulation_results = {
                "model": model,
                "times": times,
                "phases": phases,
                "order_parameter": order_parameter,
                "n_oscillators": n_oscillators,
                "frequencies": frequencies
            }
    
    # Display simulation results if available
    if "simulation_results" in st.session_state:
        results = st.session_state.simulation_results
        display_simulation_results(
            model=results["model"],
            times=results["times"],
            phases=results["phases"],
            order_parameter=results["order_parameter"],
            n_oscillators=results["n_oscillators"],
            frequencies=results["frequencies"]
        )
    else:
        # Display a welcome message for first-time users
        st.markdown("""
        <div style="background-color: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px; text-align: center;">
            <h2>Welcome to the Kuramoto Model Simulator</h2>
            <p>This interactive tool allows you to explore synchronization dynamics in coupled oscillator networks.</p>
            <p>Set your parameters in the sidebar and click "Run Simulation" to begin.</p>
            <p>The visualization will appear here after the simulation runs.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Add a quick guide
        with st.expander("Quick Guide - How to Use This Simulator"):
            st.markdown("""
            ### Getting Started
            
            1. **Set Parameters**: Adjust the parameters in the sidebar to configure your simulation:
               - Number of oscillators and coupling strength
               - Frequency distribution type and parameters
               - Simulation time and time step
               - Network connectivity
            
            2. **Run the Simulation**: Click the "Run Simulation" button to start the simulation
            
            3. **Explore Results**: After the simulation completes, explore the visualizations across the tabs:
               - Phase Visualization: See how oscillators synchronize
               - Order Parameter: Track the level of synchronization over time
               - Network Graph: Visualize the connectivity between oscillators
               - Data: Access raw simulation data
            
            ### Tips
            
            - Use the "Optimize" button for time step to automatically determine the best step size
            - Try different network types to see how topology affects synchronization
            - Save interesting simulations to the database for later reference
            - Experiment with coupling strength to find the critical threshold for synchronization
            
            ### Key Concepts
            
            - **Order Parameter**: Measures the level of synchronization (1 = perfect sync, 0 = no sync)
            - **Critical Coupling**: The threshold value of coupling strength where synchronization emerges
            - **Network Topology**: The pattern of connections between oscillators
            """)

# Database tab
with db_tab:
    render_database_tab()

# Machine Learning tab
with ml_tab:
    render_ml_tab()