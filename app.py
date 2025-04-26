"""
Kuramoto Model Simulator - Interactive visualization of coupled oscillator dynamics.
"""

import streamlit as st
import numpy as np
import json

# Import application components
from src.config.ui_config import configure_page
from src.utils.session_state import initialize_session_state, process_loaded_config, process_oscillator_count_update
from src.utils.json_handler import parse_json_parameters, update_session_from_json
from src.components.sidebar import render_sidebar
from src.components.frequencies import render_frequency_controls
from src.components.visualization import create_simulation_tabs
from src.simulation import run_simulation
from src.database.database import save_configuration, get_configuration, list_configurations

def main():
    """Main application entry point."""
    # Configure page layout and styling
    configure_page()
    
    # Initialize session state variables
    initialize_session_state()
    
    # Apply any loaded configuration
    process_loaded_config()
    
    # Process any pending oscillator count updates
    process_oscillator_count_update()
    
    # Add preset configurations dropdown at the top of the app
    configurations = list_configurations()
    if configurations and len(configurations) > 0:
        st.markdown("<h2 class='gradient_text1'>Saved Configurations</h2>", unsafe_allow_html=True)
        
        # Create columns for the dropdown and load button
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_config = st.selectbox(
                "Load a saved configuration",
                options=[f"{config['name']} (ID: {config['id']})" for config in configurations],
                index=0,
                key="config_dropdown"
            )
        
        with col2:
            if st.button("Load", key="load_config_btn"):
                # Extract config ID from the selected option
                config_id = int(selected_config.split("ID: ")[1].strip(")"))
                
                # Retrieve the configuration
                config = get_configuration(config_id)
                if config:
                    st.session_state.loaded_config = config
                    st.success(f"Loaded configuration: {config['name']}")
                    st.rerun()  # Force rerun to apply the loaded configuration
    
    # Check if we have temporary imported parameters from JSON
    if 'temp_imported_params' in st.session_state:
        update_session_from_json(st.session_state.temp_imported_params)
        # Clear the temp parameters to avoid reapplying
        del st.session_state.temp_imported_params
    
    # Add main title with gradient text
    st.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)
    
    # Render sidebar with all configuration controls
    # Pass the frequency controls function as a callback
    n_oscillators, coupling_strength, frequencies, network_type, simulation_time, random_seed, adj_matrix = render_sidebar(render_frequency_controls)
    
    # Run simulation
    model, times, phases, order_parameter = run_simulation(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        random_seed=random_seed,
        adjacency_matrix=adj_matrix
    )
    
    # Create visualization tabs
    create_simulation_tabs(
        model=model,
        times=times,
        phases=phases,
        order_parameter=order_parameter,
        n_oscillators=n_oscillators,
        network_type=network_type,
        adj_matrix=adj_matrix
    )

if __name__ == "__main__":
    main()