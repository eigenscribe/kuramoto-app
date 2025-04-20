"""
Database operations component for the Kuramoto simulator.
This module provides UI components for interacting with the database.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt

from src.database.db import (
    list_simulations, get_simulation, delete_simulation,
    list_configurations, get_configuration, delete_configuration,
    save_configuration
)

def render_database_tab():
    """
    Render the database tab with simulation and configuration management.
    
    Returns:
    --------
    dict or None
        Selected configuration if load button is clicked, otherwise None
    """
    st.markdown("## 📊 Database Management")
    
    # Create tabs for simulations and configurations
    db_tabs = st.tabs(["📈 Saved Simulations", "⚙️ Saved Configurations"])
    
    # Tab for saved simulations
    with db_tabs[0]:
        st.markdown("### Saved Simulation Results")
        st.markdown("View or delete previously run simulations.")
        
        # Get list of simulations
        simulations = list_simulations()
        
        if not simulations:
            st.info("No saved simulations found.")
        else:
            # Convert to DataFrame for better display
            sim_df = pd.DataFrame(simulations)
            
            # Format timestamp column
            sim_df['timestamp'] = pd.to_datetime(sim_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Rename columns for better UI
            sim_df = sim_df.rename(columns={
                'id': 'ID',
                'timestamp': 'Date & Time',
                'n_oscillators': 'Oscillators',
                'coupling_strength': 'Coupling',
                'simulation_time': 'Sim Time (s)',
                'frequency_distribution': 'Freq. Dist.'
            })
            
            # Display table
            st.dataframe(sim_df, use_container_width=True)
            
            # Select simulation to view or delete
            col1, col2 = st.columns(2)
            
            with col1:
                sim_id = st.selectbox(
                    "Select Simulation ID",
                    options=[sim['id'] for sim in simulations],
                    format_func=lambda x: f"ID: {x} - {next(sim['timestamp'] for sim in simulations if sim['id'] == x)}"
                )
            
            with col2:
                action = st.selectbox(
                    "Action",
                    options=["View Details", "Delete"]
                )
            
            if action == "View Details" and st.button("Execute"):
                simulation = get_simulation(sim_id)
                
                if simulation:
                    st.markdown(f"### Simulation Details (ID: {simulation['id']})")
                    
                    # Create expandable sections for different aspects
                    with st.expander("Parameters", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Oscillators", simulation['n_oscillators'])
                            st.metric("Coupling Strength", f"{simulation['coupling_strength']:.2f}")
                        
                        with col2:
                            st.metric("Simulation Time", f"{simulation['simulation_time']:.2f}s")
                            st.metric("Time Step", f"{simulation['time_step']:.3f}s")
                        
                        with col3:
                            st.metric("Random Seed", simulation['random_seed'])
                            st.metric("Frequency Dist.", simulation['frequency_distribution'])
                        
                        if simulation['frequency_params']:
                            st.json(simulation['frequency_params'])
                    
                    # Show results with tabs for different visualizations
                    st.markdown("### Results")
                    viz_tabs = st.tabs(["Order Parameter", "Phase Evolution", "Frequency Distribution"])
                    
                    with viz_tabs[0]:
                        # Create order parameter plot
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # We only have samples from the order parameter
                        # Create a simple time array based on simulation parameters
                        times = np.linspace(0, simulation['simulation_time'], len(simulation['order_parameter']))
                        
                        ax.plot(times, simulation['order_parameter'], linewidth=2)
                        ax.fill_between(times, 0, simulation['order_parameter'], alpha=0.3)
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Order Parameter (r)')
                        ax.set_title('Synchronization over Time')
                        ax.set_ylim(0, 1.05)
                        ax.grid(alpha=0.3)
                        
                        st.pyplot(fig)
                        
                        # Display final synchronization value
                        final_r = simulation['order_parameter'][-1]
                        st.metric("Final Synchronization", f"{final_r:.3f}")
                    
                    with viz_tabs[1]:
                        # Phase evolution is typically stored sparsely to save space
                        # We'll create a simplified visualization
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Create times array
                        times = np.linspace(0, simulation['simulation_time'], simulation['phases'].shape[1])
                        
                        # Plot phases for each oscillator
                        for i in range(simulation['n_oscillators']):
                            ax.plot(times, simulation['phases'][i, :] % (2 * np.pi), alpha=0.7, lw=1)
                        
                        ax.set_xlabel('Time')
                        ax.set_ylabel('Phase (mod 2π)')
                        ax.set_title('Oscillator Phases Over Time')
                        ax.set_ylim(0, 2 * np.pi)
                        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
                        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
                        ax.grid(alpha=0.3)
                        
                        st.pyplot(fig)
                    
                    with viz_tabs[2]:
                        # Create frequency histogram
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        bins = min(20, len(simulation['frequencies']))
                        ax.hist(simulation['frequencies'], bins=bins, alpha=0.7, density=True)
                        ax.set_xlabel('Natural Frequency (ω)')
                        ax.set_ylabel('Probability Density')
                        ax.set_title(f"Frequency Distribution ({simulation['frequency_distribution']})")
                        ax.grid(alpha=0.3)
                        
                        # Add statistics as text
                        mean = np.mean(simulation['frequencies'])
                        std = np.std(simulation['frequencies'])
                        min_freq = np.min(simulation['frequencies'])
                        max_freq = np.max(simulation['frequencies'])
                        
                        stats_text = f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\nMin: {min_freq:.3f}\nMax: {max_freq:.3f}'
                        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
                        
                        st.pyplot(fig)
                    
                    # Allow saving this simulation as a configuration
                    if st.button("Save as Configuration"):
                        config_name = f"Config from Sim {simulation['id']}"
                        
                        # Determine network type from adjacency matrix
                        network_type = "Custom" if simulation['adjacency_matrix'] is not None else "All-to-All"
                        
                        # Save the configuration
                        save_configuration(
                            name=config_name,
                            n_oscillators=simulation['n_oscillators'],
                            coupling_strength=simulation['coupling_strength'],
                            simulation_time=simulation['simulation_time'],
                            time_step=simulation['time_step'],
                            random_seed=simulation['random_seed'],
                            network_type=network_type,
                            frequency_distribution=simulation['frequency_distribution'],
                            frequency_params=simulation['frequency_params'],
                            adjacency_matrix=simulation['adjacency_matrix']
                        )
                        
                        st.success(f"Saved as configuration: {config_name}")
                else:
                    st.error("Simulation not found.")
            
            elif action == "Delete" and st.button("Execute", type="primary"):
                if delete_simulation(sim_id):
                    st.success(f"Simulation {sim_id} deleted successfully.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to delete simulation {sim_id}.")
    
    # Tab for saved configurations
    with db_tabs[1]:
        st.markdown("### Saved Configurations")
        st.markdown("Load or manage saved simulation configurations.")
        
        # Get list of configurations
        configs = list_configurations()
        
        # Variable to store the selected configuration
        selected_config = None
        
        if not configs:
            st.info("No saved configurations found.")
        else:
            # Convert to DataFrame for better display
            config_df = pd.DataFrame(configs)
            
            # Format timestamp column
            config_df['timestamp'] = pd.to_datetime(config_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Rename columns for better UI
            config_df = config_df.rename(columns={
                'id': 'ID',
                'name': 'Name',
                'timestamp': 'Date & Time',
                'n_oscillators': 'Oscillators',
                'coupling_strength': 'Coupling',
                'network_type': 'Network Type',
                'frequency_distribution': 'Freq. Dist.'
            })
            
            # Display table
            st.dataframe(config_df, use_container_width=True)
            
            # Select configuration to view, load, or delete
            col1, col2 = st.columns(2)
            
            with col1:
                config_id = st.selectbox(
                    "Select Configuration",
                    options=[config['id'] for config in configs],
                    format_func=lambda x: f"{next(config['name'] for config in configs if config['id'] == x)}"
                )
            
            with col2:
                action = st.selectbox(
                    "Action",
                    options=["View Details", "Load", "Delete"]
                )
            
            if action == "View Details" and st.button("Execute"):
                config = get_configuration(config_id)
                
                if config:
                    st.markdown(f"### Configuration Details: {config['name']}")
                    
                    # Create expandable section for parameters
                    with st.expander("Parameters", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Oscillators", config['n_oscillators'])
                            st.metric("Coupling Strength", f"{config['coupling_strength']:.2f}")
                        
                        with col2:
                            st.metric("Simulation Time", f"{config['simulation_time']:.2f}s")
                            st.metric("Time Step", f"{config['time_step']:.3f}s")
                        
                        with col3:
                            st.metric("Network Type", config['network_type'])
                            st.metric("Frequency Dist.", config['frequency_distribution'])
                        
                        st.metric("Random Seed", config['random_seed'])
                        
                        if config['frequency_params']:
                            st.subheader("Frequency Parameters")
                            st.json(config['frequency_params'])
                    
                    # Show network structure if applicable
                    if config['adjacency_matrix'] is not None:
                        st.subheader("Network Structure")
                        
                        # Create a plot of the network
                        fig, ax = plt.subplots(figsize=(8, 8))
                        
                        import networkx as nx
                        G = nx.from_numpy_array(config['adjacency_matrix'])
                        
                        # Create a circular layout
                        pos = nx.circular_layout(G)
                        
                        # Draw nodes and edges
                        nx.draw_networkx_nodes(G, pos, node_size=300, node_color='skyblue', edgecolors='white')
                        nx.draw_networkx_edges(G, pos, alpha=0.5)
                        
                        # Remove axis
                        ax.set_axis_off()
                        ax.set_title("Network Adjacency Matrix Visualization")
                        
                        st.pyplot(fig)
                else:
                    st.error("Configuration not found.")
            
            elif action == "Load" and st.button("Execute", type="primary"):
                config = get_configuration(config_id)
                if config:
                    st.success(f"Configuration '{config['name']}' loaded successfully.")
                    selected_config = config
                else:
                    st.error("Configuration not found.")
            
            elif action == "Delete" and st.button("Execute", type="primary"):
                if delete_configuration(config_id):
                    st.success(f"Configuration deleted successfully.")
                    st.experimental_rerun()
                else:
                    st.error(f"Failed to delete configuration.")
    
    return selected_config