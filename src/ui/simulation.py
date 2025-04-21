"""
Simulation UI components for Kuramoto Model Simulator.
"""
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
import time
import networkx as nx
import matplotlib.collections as mcoll

from src.models.kuramoto_model import KuramotoModel
from src.utils.database import store_simulation

def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step, random_seed, 
                  adjacency_matrix=None, auto_optimize=False, safety_factor=0.8):
    """
    Run a Kuramoto model simulation with the specified parameters and return the results.
    
    Parameters:
    -----------
    n_oscillators : int
        Number of oscillators
    coupling_strength : float
        Coupling strength parameter K
    frequencies : ndarray
        Natural frequencies of oscillators
    simulation_time : float
        Total simulation time
    time_step : float
        Simulation time step
    random_seed : int
        Seed for random number generation
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix defining network connectivity
    auto_optimize : bool, optional
        Whether to automatically optimize the time step before running the simulation
    safety_factor : float, optional
        Safety factor for time step optimization (0-1, lower is more conservative)
        
    Returns:
    --------
    tuple
        (model, times, phases, order_parameter, optimized_time_step)
        Note: optimized_time_step will be None if auto_optimize=False
    """
    # Initialize model
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed,
        adjacency_matrix=adjacency_matrix
    )
    
    # Optimize time step if requested
    optimized_time_step = None
    if auto_optimize:
        optimization_results = model.set_optimal_time_step(safety_factor=safety_factor)
        optimized_time_step = optimization_results['optimal_time_step']
        print(f"Optimized time step: {optimized_time_step}")
    
    # Run the simulation
    times, phases, order_parameter = model.simulate()
    
    return model, times, phases, order_parameter, optimized_time_step

def display_simulation_results(model, times, phases, order_parameter, n_oscillators, frequencies):
    """Display simulation results in the main content area."""
    # Format phases for display (unwrap to show continuous rotation beyond 2π)
    phases_unwrapped = np.unwrap(phases, axis=0)
    
    # Calculate the mean phase (unwrapped) for each time point
    mean_phases = np.mean(phases_unwrapped, axis=1)
    
    # Calculate the synchronized frequency
    if len(times) > 1 and len(mean_phases) > 1:
        # Make sure arrays have the same shape
        min_len = min(len(mean_phases), len(times))
        if min_len > 1:
            phase_diff = np.diff(mean_phases[:min_len])
            time_diff = np.diff(times[:min_len])
            sync_frequency = np.mean(phase_diff / time_diff)
        else:
            sync_frequency = 0.0
    else:
        sync_frequency = 0.0
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Phase Visualization", "Order Parameter", "Network Graph", "Data"])
    
    with tab1:
        st.markdown("<h3 class='gradient_text1'>Phase Dynamics</h3>", unsafe_allow_html=True)
        
        # Create three columns for the plots
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Initial Oscillator Positions")
            
            # Create a figure for initial positions
            fig_initial = plt.figure(figsize=(5, 5))
            ax_initial = fig_initial.add_subplot(111, projection='polar')
            
            # Make sure the arrays are the same length
            if len(phases[0]) == n_oscillators:
                ax_initial.scatter(phases[0], np.ones(n_oscillators), c=frequencies, s=80, alpha=0.8, cmap='viridis')
            else:
                # If there's a mismatch, adjust the arrays to match
                min_len = min(len(phases[0]), n_oscillators)
                ax_initial.scatter(phases[0][:min_len], np.ones(min_len), 
                                  c=frequencies[:min_len] if len(frequencies) >= min_len else frequencies, 
                                  s=80, alpha=0.8, cmap='viridis')
            
            ax_initial.set_rticks([])  # Hide radial ticks
            ax_initial.set_title("t=0")
            
            # Plot mean phase as a red line
            mean_phase_0 = np.mean(phases[0])
            ax_initial.plot([mean_phase_0, mean_phase_0], [0, 1], 'r-', lw=2)
            
            st.pyplot(fig_initial)
        
        with col2:
            st.subheader("Final Oscillator Positions")
            
            # Create a figure for final positions
            fig_final = plt.figure(figsize=(5, 5))
            ax_final = fig_final.add_subplot(111, projection='polar')
            
            # Make sure the arrays are the same length
            if len(phases[-1]) == n_oscillators:
                ax_final.scatter(phases[-1], np.ones(n_oscillators), c=frequencies, s=80, alpha=0.8, cmap='viridis')
            else:
                # If there's a mismatch, adjust the arrays to match
                min_len = min(len(phases[-1]), n_oscillators)
                ax_final.scatter(phases[-1][:min_len], np.ones(min_len), 
                               c=frequencies[:min_len] if len(frequencies) >= min_len else frequencies, 
                               s=80, alpha=0.8, cmap='viridis')
                
            ax_final.set_rticks([])  # Hide radial ticks
            ax_final.set_title(f"t={times[-1]:.1f}")
            
            # Plot mean phase as a red line
            mean_phase_final = np.mean(phases[-1])
            ax_final.plot([mean_phase_final, mean_phase_final], [0, 1], 'r-', lw=2)
            
            st.pyplot(fig_final)
        
        with col3:
            # Create an interactive time slider for viewing phases at different times
            st.subheader("Phase Development Over Time")
            
            # Ensure we have a valid number of timepoints
            n_time_points = len(times)
            
            # Default the slider to the first frame (0 index)
            # We already initialize this in common.py to 0
            
            # Ensure the stored time_idx is valid
            # Reset to 0 after a new simulation is run
            if "new_simulation_run" in st.session_state and st.session_state.new_simulation_run:
                st.session_state.time_idx = 0
                st.session_state.new_simulation_run = False
            else:
                st.session_state.time_idx = min(max(0, st.session_state.time_idx), max(0, n_time_points - 1))
            
            # Create the slider with fail-safe values
            time_idx = st.slider(
                "Time",
                min_value=0,
                max_value=max(0, n_time_points - 1),  # Ensure we don't get negative slider values
                value=st.session_state.time_idx,
                format="t=%d",
                key="time_idx"
            )
            
            # Show actual time value (with bounds checking)
            if 0 <= time_idx < n_time_points:
                st.markdown(f"*t = {times[time_idx]:.2f}*")
            else:
                st.markdown("*Invalid time index*")
            
            # Create a figure for the selected time
            fig_selected = plt.figure(figsize=(5, 5))
            ax_selected = fig_selected.add_subplot(111, projection='polar')
            
            # Ensure time_idx is within bounds
            if 0 <= time_idx < len(phases):
                # Make sure the arrays are the same length
                if len(phases[time_idx]) == n_oscillators:
                    scatter = ax_selected.scatter(phases[time_idx], np.ones(n_oscillators), 
                                                 c=frequencies, s=80, alpha=0.8, cmap='viridis')
                else:
                    # If there's a mismatch, adjust the arrays to match
                    min_len = min(len(phases[time_idx]), n_oscillators)
                    scatter = ax_selected.scatter(phases[time_idx][:min_len], np.ones(min_len), 
                                                c=frequencies[:min_len] if len(frequencies) >= min_len else frequencies, 
                                                s=80, alpha=0.8, cmap='viridis')
                
                ax_selected.set_rticks([])  # Hide radial ticks
                
                # Plot mean phase as a red line
                mean_phase = np.mean(phases[time_idx])
                ax_selected.plot([mean_phase, mean_phase], [0, 1], 'r-', lw=2)
                
                # Add a colorbar to indicate the natural frequencies
                cbar = fig_selected.colorbar(scatter, ax=ax_selected, shrink=0.8)
                cbar.set_label('Natural Frequency')
            else:
                # Display an error message on the plot if time_idx is invalid
                ax_selected.text(0, 0, "No data to display", 
                                ha='center', va='center', fontsize=12)
                ax_selected.set_rticks([])
            
            st.pyplot(fig_selected)
        
        # Animation section
        st.markdown("<h3 class='gradient_text1'>Phase Animation</h3>", unsafe_allow_html=True)
        
        # Initialize animation state in session
        if 'animation_playing' not in st.session_state:
            st.session_state.animation_playing = False
        if 'animation_speed' not in st.session_state:
            st.session_state.animation_speed = 1.0
        
        # Create two columns for animation controls
        anim_col1, anim_col2 = st.columns([1, 3])
        
        with anim_col1:
            # Play/Pause button
            if st.button("▶️ Play" if not st.session_state.animation_playing else "⏸️ Pause", 
                         key="play_pause_button"):
                st.session_state.animation_playing = not st.session_state.animation_playing
        
        with anim_col2:
            # Animation speed slider
            animation_speed = st.slider(
                "Animation Speed",
                min_value=0.25,
                max_value=3.0,
                value=st.session_state.animation_speed,
                step=0.25,
                format="%.2fx",
                key="animation_speed"
            )
        
        # Create a container for the animation
        animation_container = st.empty()
        
        # Function to create animation frames
        def create_animation_frame(i):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar')
            
            # Check if index is valid
            if i < 0 or i >= len(phases):
                # Display an error message if the index is invalid
                ax.text(0, 0, "Invalid time index", 
                      ha='center', va='center', fontsize=14)
                ax.set_rticks([])  # Hide radial ticks
                
                # Save the figure to a BytesIO object
                buf = BytesIO()
                fig.savefig(buf, format='png', bbox_inches='tight')
                buf.seek(0)
                plt.close(fig)
                return base64.b64encode(buf.read()).decode('utf-8')
            
            # Get the current phase values
            current_phases = phases[i]
            
            # Make sure the arrays are the same length
            if len(current_phases) == n_oscillators:
                scatter = ax.scatter(current_phases, np.ones(n_oscillators), 
                                   c=frequencies, cmap='viridis', 
                                   s=150, alpha=0.8)
            else:
                # If there's a mismatch, adjust the arrays to match
                min_len = min(len(current_phases), n_oscillators)
                scatter = ax.scatter(current_phases[:min_len], np.ones(min_len), 
                                   c=frequencies[:min_len] if len(frequencies) >= min_len else frequencies, 
                                   s=150, alpha=0.8, cmap='viridis')
            
            # Plot mean phase as a red line
            mean_phase = np.mean(current_phases)
            ax.plot([mean_phase, mean_phase], [0, 1], 'r-', lw=2)
            
            # Set up the plot
            ax.set_rticks([])  # Hide radial ticks
            
            # Make sure times index is also valid
            if i < len(times):
                ax.set_title(f"t = {times[i]:.2f}", fontsize=16)
            else:
                ax.set_title("Time not available", fontsize=16)
            
            # Add a colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Natural Frequency', fontsize=12)
            
            # Add order parameter text (with bounds checking)
            if i < len(order_parameter):
                ax.text(0.5, -0.1, f"Order Parameter r = {order_parameter[i]:.3f}", 
                      transform=ax.transAxes, ha='center', fontsize=14)
            
            # Save the figure to a BytesIO object
            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            plt.close(fig)
            
            # Return the image as base64 encoded string
            return base64.b64encode(buf.read()).decode('utf-8')
        
        # Display the animation if it's playing
        if st.session_state.animation_playing:
            # Calculate appropriate frame rate based on animation speed
            frames_to_show = min(60, len(times))  # Limit to 60 frames for performance
            frame_interval = max(1, len(times) // frames_to_show)
            
            # Get the selected time index for resuming
            start_idx = min(time_idx, len(times) - frames_to_show * frame_interval)
            
            # Calculate frames to display
            frame_indices = range(start_idx, len(times), frame_interval)
            
            # Safely get frame indices
            if len(frame_indices) > 0:
                # Create and display first frame
                first_frame_base64 = create_animation_frame(frame_indices[0])
                img_html = f'<img src="data:image/png;base64,{first_frame_base64}" style="width: 100%;">'
                animation_container.markdown(img_html, unsafe_allow_html=True)
                
                # Animate subsequent frames
                for i in frame_indices[1:]:
                    if not st.session_state.animation_playing:
                        break
                    
                    # Create and update the frame
                    frame_base64 = create_animation_frame(i)
                    img_html = f'<img src="data:image/png;base64,{frame_base64}" style="width: 100%;">'
                    animation_container.markdown(img_html, unsafe_allow_html=True)
                    
                    # Control animation speed
                    time.sleep(0.1 / st.session_state.animation_speed)
            else:
                # No valid frames to show
                animation_container.warning("No frames available for animation")
                
            # Once animation is done, update playing state
            st.session_state.animation_playing = False
        else:
            # Display static frame at current time index (make sure it's valid)
            if 0 <= time_idx < len(times):
                static_frame_base64 = create_animation_frame(time_idx)
                img_html = f'<img src="data:image/png;base64,{static_frame_base64}" style="width: 100%;">'
                animation_container.markdown(img_html, unsafe_allow_html=True)
            else:
                animation_container.warning("No valid frame to display")
        
        # Phase trajectory plot (unwrapped phases over time)
        st.markdown("<h3 class='gradient_text1'>Phase Trajectories</h3>", unsafe_allow_html=True)
        
        phase_fig, phase_ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual phase trajectories with color based on natural frequency
        norm = plt.Normalize(np.min(frequencies), np.max(frequencies))
        cmap = plt.cm.viridis
        
        for i in range(n_oscillators):
            color = cmap(norm(frequencies[i]))
            phase_ax.plot(times, phases_unwrapped[:, i], color=color, alpha=0.7, lw=1.5)
        
        # Plot mean phase trajectory
        phase_ax.plot(times, mean_phases, 'r-', lw=2.5, label='Mean Phase')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=phase_ax)
        cbar.set_label('Natural Frequency')
        
        # Add labels and legend
        phase_ax.set_xlabel('Time')
        phase_ax.set_ylabel('Phase (unwrapped)')
        phase_ax.set_title('Oscillator Phase Trajectories')
        phase_ax.legend()
        phase_ax.grid(True, alpha=0.3)
        
        st.pyplot(phase_fig)
        
        # Phase difference visualization
        st.markdown("<h3 class='gradient_text1'>Phase Differences</h3>", unsafe_allow_html=True)
        
        # Calculate phase differences relative to the mean phase
        phase_diffs = phases_unwrapped - mean_phases[:, np.newaxis]
        
        diff_fig, diff_ax = plt.subplots(figsize=(10, 6))
        
        # Plot phase differences with color based on natural frequency
        for i in range(n_oscillators):
            color = cmap(norm(frequencies[i]))
            diff_ax.plot(times, phase_diffs[:, i], color=color, alpha=0.7, lw=1.5)
        
        # Add reference line at zero
        diff_ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=diff_ax)
        cbar.set_label('Natural Frequency')
        
        # Add labels
        diff_ax.set_xlabel('Time')
        diff_ax.set_ylabel('Phase Difference from Mean')
        diff_ax.set_title('Phase Differences Relative to Mean Phase')
        diff_ax.grid(True, alpha=0.3)
        
        st.pyplot(diff_fig)
    
    with tab2:
        st.markdown("<h3 class='gradient_text1'>Order Parameter</h3>", unsafe_allow_html=True)
        
        # Create columns for the plots
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Order parameter over time
            r_fig, r_ax = plt.subplots(figsize=(10, 6))
            r_ax.plot(times, order_parameter, 'b-', lw=2.5)
            r_ax.set_xlabel('Time')
            r_ax.set_ylabel('Order Parameter (r)')
            r_ax.set_title('Synchronization Level Over Time')
            r_ax.grid(True, alpha=0.3)
            
            # Add horizontal line for final value
            r_ax.axhline(y=order_parameter[-1], color='r', linestyle='--', 
                        label=f'Final r = {order_parameter[-1]:.3f}')
            r_ax.legend()
            
            st.pyplot(r_fig)
        
        with col2:
            # Summary statistics
            st.markdown("<h4>Synchronization Summary</h4>", unsafe_allow_html=True)
            
            # Calculate statistics
            r_mean = np.mean(order_parameter)
            r_final = order_parameter[-1]
            r_max = np.max(order_parameter)
            
            # Determine if system reached synchronization
            sync_threshold = 0.9
            is_synchronized = r_final > sync_threshold
            
            # Mean natural frequency and standard deviation
            mean_freq = np.mean(frequencies)
            std_freq = np.std(frequencies)
            
            # Calculate the critical coupling based on frequency distribution
            critical_k = 2 * std_freq / np.pi
            
            # Create a card with the statistics
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.2); padding: 20px; border-radius: 10px;">
                <h5>Order Parameter</h5>
                <p>Final value: <b>{r_final:.3f}</b></p>
                <p>Mean value: {r_mean:.3f}</p>
                <p>Max value: {r_max:.3f}</p>
                
                <h5>Synchronization</h5>
                <p>Status: <span style="color: {'green' if is_synchronized else 'orange'};">
                    <b>{'Synchronized' if is_synchronized else 'Partially Synchronized'}</b>
                </span></p>
                <p>Synchronized frequency: {sync_frequency:.3f}</p>
                
                <h5>Critical Coupling</h5>
                <p>Theoretical Kc: {critical_k:.3f}</p>
                <p>K/Kc ratio: {model.coupling_strength / critical_k if critical_k > 0 else 'N/A':.2f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Add a histogram of natural frequencies
            st.markdown("<h4>Natural Frequency Distribution</h4>", unsafe_allow_html=True)
            
            freq_fig, freq_ax = plt.subplots(figsize=(8, 4))
            freq_ax.hist(frequencies, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
            freq_ax.axvline(x=mean_freq, color='red', linestyle='--', 
                          label=f'Mean: {mean_freq:.2f}')
            freq_ax.set_xlabel('Natural Frequency')
            freq_ax.set_ylabel('Count')
            freq_ax.legend()
            freq_ax.grid(True, alpha=0.3)
            
            st.pyplot(freq_fig)
    
    with tab3:
        st.markdown("<h3 class='gradient_text1'>Network Visualization</h3>", unsafe_allow_html=True)
        
        # Create network graph from adjacency matrix or default network type
        if hasattr(model, 'adjacency_matrix') and model.adjacency_matrix is not None:
            adj_matrix = model.adjacency_matrix
            
            # Create network graph
            G = nx.from_numpy_array(adj_matrix)
            
            # Display network information
            st.markdown(f"""
            <div style="background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 10px; margin-bottom: 15px;">
                <h4>Network Information</h4>
                <p>Nodes: {G.number_of_nodes()}</p>
                <p>Edges: {G.number_of_edges()}</p>
                <p>Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}</p>
                <p>Density: {nx.density(G):.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create two columns for different visualizations
            net_col1, net_col2 = st.columns(2)
            
            with net_col1:
                st.markdown("<h4>Network Structure</h4>", unsafe_allow_html=True)
                
                # Create network plot
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Use different layout algorithms depending on network size
                if n_oscillators <= 20:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42)  # Use a default seed
                
                # Create a colormap based on oscillator frequencies
                node_colors = frequencies
                
                # Draw the network
                nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                                     node_size=300, alpha=0.8, cmap='viridis',
                                     ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.5, ax=ax)
                
                # Add node labels if network is small enough
                if n_oscillators <= 20:
                    nx.draw_networkx_labels(G, pos, font_color='white', 
                                         font_size=10, ax=ax)
                
                # Add a colorbar
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                         norm=plt.Normalize(vmin=min(frequencies), 
                                                          vmax=max(frequencies)))
                sm.set_array([])
                plt.colorbar(sm, ax=ax, label='Natural Frequency')
                
                # Set title and remove axis
                ax.set_title("Network Connectivity")
                ax.axis('off')
                
                st.pyplot(fig)
                
            with net_col2:
                st.markdown("<h4>Final Phase Visualization</h4>", unsafe_allow_html=True)
                
                # Create network plot with phase information
                fig, ax = plt.subplots(figsize=(8, 8))
                
                # Use the same layout as before
                if n_oscillators <= 20:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42)  # Use fixed seed for consistency
                
                # Create a colormap for phases
                phase_colors = phases[-1]
                
                # Create arrow properties based on phases
                arrow_sizes = np.ones(n_oscillators) * 0.2
                
                # Draw the network nodes
                nx.draw_networkx_nodes(G, pos, 
                                     node_color=node_colors, 
                                     node_size=300, 
                                     alpha=0.6, 
                                     cmap='viridis',
                                     ax=ax)
                
                # Draw the edges
                nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
                
                # Draw arrows representing phases
                for i, (node, position) in enumerate(pos.items()):
                    phase = phases[-1][i]
                    dx = arrow_sizes[i] * np.cos(phase)
                    dy = arrow_sizes[i] * np.sin(phase)
                    
                    ax.arrow(position[0], position[1], 
                           dx, dy, 
                           head_width=0.03, 
                           head_length=0.05, 
                           fc=plt.cm.hsv(phase/(2*np.pi)), 
                           ec='black', 
                           alpha=0.8)
                
                # Add node labels if network is small enough
                if n_oscillators <= 20:
                    nx.draw_networkx_labels(G, pos, font_color='white', 
                                         font_size=10, ax=ax)
                
                # Set title and remove axis
                ax.set_title("Final Oscillator Phases")
                ax.axis('off')
                
                # Add a colorbar for phases
                sm_phase = plt.cm.ScalarMappable(cmap=plt.cm.hsv, 
                                              norm=plt.Normalize(vmin=0, vmax=2*np.pi))
                sm_phase.set_array([])
                plt.colorbar(sm_phase, ax=ax, label='Phase')
                
                st.pyplot(fig)
            
            # Visualization of phase evolution on the network
            st.markdown("<h4>Network Phase Evolution</h4>", unsafe_allow_html=True)
            
            # Select time points to visualize
            n_frames = 3  # Number of snapshots to show
            time_points = np.linspace(0, len(times)-1, n_frames, dtype=int)
            
            # Create a figure with subplots for each time point
            fig, axes = plt.subplots(1, n_frames, figsize=(15, 5))
            
            # Loop through each time point and create a visualization
            for i, t_idx in enumerate(time_points):
                ax = axes[i]
                
                # Use the same layout as before
                if n_oscillators <= 20:
                    pos = nx.circular_layout(G)
                else:
                    pos = nx.spring_layout(G, seed=42)  # Use fixed seed for consistency
                
                # Draw the network structure
                nx.draw_networkx_nodes(G, pos, 
                                     node_color=[plt.cm.hsv(phases[t_idx][j]/(2*np.pi)) for j in range(n_oscillators)],
                                     node_size=300, 
                                     alpha=0.7,
                                     ax=ax)
                nx.draw_networkx_edges(G, pos, alpha=0.3, ax=ax)
                
                # Draw arrows representing phases
                for j, (node, position) in enumerate(pos.items()):
                    phase = phases[t_idx][j]
                    dx = 0.2 * np.cos(phase)
                    dy = 0.2 * np.sin(phase)
                    
                    ax.arrow(position[0], position[1], 
                           dx, dy, 
                           head_width=0.03, 
                           head_length=0.05, 
                           fc=plt.cm.hsv(phase/(2*np.pi)), 
                           ec='black', 
                           alpha=0.8)
                
                # Add title with time and order parameter
                ax.set_title(f"t={times[t_idx]:.1f}\nr={order_parameter[t_idx]:.2f}")
                ax.axis('off')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    with tab4:
        st.markdown("<h3 class='gradient_text1'>Simulation Data</h3>", unsafe_allow_html=True)
        
        # Create two columns for data and controls
        data_col1, data_col2 = st.columns([3, 2])
        
        with data_col1:
            # Display simulation parameters
            st.markdown("<h4>Simulation Parameters</h4>", unsafe_allow_html=True)
            
            param_data = {
                "Number of Oscillators": n_oscillators,
                "Coupling Strength (K)": model.coupling_strength,
                "Simulation Time": model.simulation_time,
                "Time Step": model.time_step,
                "Random Seed": model.random_seed if hasattr(model, 'random_seed') else 42,
                "Network Type": model.network_type if hasattr(model, 'network_type') else "Custom",
                "Mean Natural Frequency": np.mean(frequencies),
                "Std. Dev. of Natural Frequencies": np.std(frequencies)
            }
            
            # Format as a Bootstrap-style table
            param_table_html = """
            <div style="background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 15px;">
            <table class="table">
                <thead>
                    <tr>
                        <th style="width: 50%;">Parameter</th>
                        <th style="width: 50%;">Value</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for param, value in param_data.items():
                # Format numerical values to 4 decimal places
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                param_table_html += f"""
                <tr>
                    <td>{param}</td>
                    <td>{value_str}</td>
                </tr>
                """
            
            param_table_html += """
                </tbody>
            </table>
            </div>
            """
            
            st.markdown(param_table_html, unsafe_allow_html=True)
            
            # Display results summary
            st.markdown("<h4>Results Summary</h4>", unsafe_allow_html=True)
            
            results_data = {
                "Final Order Parameter": order_parameter[-1],
                "Maximum Order Parameter": np.max(order_parameter),
                "Mean Order Parameter": np.mean(order_parameter),
                "Synchronized Frequency": sync_frequency,
                "Frequency Range": f"{np.min(frequencies):.4f} to {np.max(frequencies):.4f}",
                "Critical Coupling Estimate": f"{2 * np.std(frequencies) / np.pi:.4f}",
                "Computation Time": f"{model.computation_time:.4f} seconds" if hasattr(model, 'computation_time') else "Not recorded"
            }
            
            # Format as a Bootstrap-style table
            results_table_html = """
            <div style="background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 15px;">
            <table class="table">
                <thead>
                    <tr>
                        <th style="width: 50%;">Metric</th>
                        <th style="width: 50%;">Value</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            for metric, value in results_data.items():
                if isinstance(value, float):
                    value_str = f"{value:.4f}"
                else:
                    value_str = str(value)
                
                results_table_html += f"""
                <tr>
                    <td>{metric}</td>
                    <td>{value_str}</td>
                </tr>
                """
            
            results_table_html += """
                </tbody>
            </table>
            </div>
            """
            
            st.markdown(results_table_html, unsafe_allow_html=True)
        
        with data_col2:
            # Create a section for saving the results to the database
            st.markdown("<h4>Save Simulation Results</h4>", unsafe_allow_html=True)
            
            # Input field for simulation name
            simulation_name = st.text_input(
                "Simulation Name",
                value=f"Simulation_N{n_oscillators}_K{model.coupling_strength}",
                help="Enter a name for this simulation to save it to the database"
            )
            
            # Input field for notes
            simulation_notes = st.text_area(
                "Notes",
                value="",
                help="Enter any notes about this simulation"
            )
            
            # Checkbox for including full phase data
            include_phases = st.checkbox(
                "Include Full Phase Data",
                value=False,
                help="Whether to include full phase data in the saved results (increases storage size)"
            )
            
            # Checkbox for including adjacency matrix
            include_adj_matrix = st.checkbox(
                "Include Adjacency Matrix",
                value=True,
                help="Whether to include the adjacency matrix in the saved results"
            )
            
            # Save button
            if st.button("Save to Database", type="primary"):
                try:
                    # Create a dictionary for frequency params
                    freq_params = {}
                    
                    # Populate frequency parameters based on distribution type
                    if st.session_state.freq_type == "Normal":
                        freq_params = {
                            "mean": st.session_state.freq_mean,
                            "std": st.session_state.freq_std
                        }
                    elif st.session_state.freq_type == "Uniform":
                        freq_params = {
                            "min": st.session_state.freq_min,
                            "max": st.session_state.freq_max
                        }
                    elif st.session_state.freq_type == "Bimodal":
                        freq_params = {
                            "peak1": st.session_state.peak1,
                            "peak2": st.session_state.peak2
                        }
                    elif st.session_state.freq_type == "Custom":
                        # Parse the custom frequencies
                        try:
                            custom_values = [float(x.strip()) for x in st.session_state.custom_freqs.split(',')]
                            freq_params = {
                                "custom_values": custom_values
                            }
                        except:
                            freq_params = {}
                    
                    # Get adjacency matrix if needed
                    adj_matrix_to_save = None
                    if include_adj_matrix and hasattr(model, 'adjacency_matrix') and model.adjacency_matrix is not None:
                        adj_matrix_to_save = model.adjacency_matrix
                    
                    # Get phase data if needed
                    phase_data_to_save = None
                    if include_phases:
                        phase_data_to_save = phases
                    
                    # Save to database
                    sim_id = store_simulation(
                        name=simulation_name,
                        description=simulation_notes,
                        n_oscillators=n_oscillators,
                        coupling_strength=model.coupling_strength,
                        frequencies=frequencies,
                        frequency_distribution=st.session_state.freq_type,
                        frequency_params=json.dumps(freq_params) if freq_params else None,
                        simulation_time=model.simulation_time,
                        time_step=model.time_step,
                        random_seed=model.random_seed if hasattr(model, 'random_seed') else 42,
                        network_type=model.network_type if hasattr(model, 'network_type') else "Custom",
                        adjacency_matrix=adj_matrix_to_save if include_adj_matrix else None,
                        order_parameter=order_parameter,
                        phases=phase_data_to_save,
                        final_sync=order_parameter[-1],
                        computation_time=model.computation_time if hasattr(model, 'computation_time') else None
                    )
                    
                    st.success(f"Simulation saved to database with ID: {sim_id}")
                    
                except Exception as e:
                    st.error(f"Error saving simulation: {str(e)}")
            
            # Add an expander with info about the database
            with st.expander("About Data Storage"):
                st.markdown("""
                Simulations saved to the database can be retrieved for later analysis or comparison.
                The database stores:
                
                - Simulation parameters
                - Order parameter time series
                - Natural frequencies
                - Final synchronization value
                - Optional: Full phase data
                - Optional: Adjacency matrix
                
                You can access and manage saved simulations in the Database tab.
                """)
                
            # Download data section
            st.markdown("<h4>Download Data</h4>", unsafe_allow_html=True)
            
            # Create CSV data for frequencies
            freq_csv = "oscillator_id,natural_frequency\n"
            for i, freq in enumerate(frequencies):
                freq_csv += f"{i},{freq}\n"
            
            # Create CSV data for order parameter
            order_csv = "time,order_parameter\n"
            for t, r in zip(times, order_parameter):
                order_csv += f"{t},{r}\n"
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                st.download_button(
                    label="Download Frequencies",
                    data=freq_csv,
                    file_name="natural_frequencies.csv",
                    mime="text/csv"
                )
            
            with col2:
                st.download_button(
                    label="Download Order Parameter",
                    data=order_csv,
                    file_name="order_parameter.csv",
                    mime="text/csv"
                )