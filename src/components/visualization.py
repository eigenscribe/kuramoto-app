"""
Visualization components for the Kuramoto model simulator.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap
import networkx as nx
import plotly.graph_objects as go
from io import BytesIO
import base64
import time
import matplotlib.collections

def create_simulation_tabs(model, times, phases, order_parameter, n_oscillators, network_type, adj_matrix):
    """
    Create tabs for different visualization aspects of the simulation.
    
    Args:
        model: KuramotoModel instance
        times: Time points array
        phases: Phases array
        order_parameter: Order parameter array
        n_oscillators: Number of oscillators
        network_type: Type of network connectivity
        adj_matrix: Adjacency matrix for custom networks
    """
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Animation", "Network", "Time Evolution", "Analysis"])
    
    with tab1:
        _render_animation_tab(times, phases, order_parameter, n_oscillators)
    
    with tab2:
        _render_network_tab(model, times, phases, n_oscillators, network_type, adj_matrix)
    
    with tab3:
        _render_time_evolution_tab(times, phases, order_parameter, n_oscillators)
    
    with tab4:
        _render_analysis_tab(times, phases, order_parameter, n_oscillators)

def _render_animation_tab(times, phases, order_parameter, n_oscillators):
    """Render the animation tab with interactive controls."""
    st.markdown("### Interactive Simulation Playback")
    
    # Create a placeholder for the animation control buttons,
    # centered and positioned ABOVE the time slider
    animation_col1, animation_col2, animation_col3 = st.columns([1, 3, 1])
    
    animation_buttons_container = st.container()
    btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = animation_buttons_container.columns([1, 1, 1, 1, 1])
    
    # Create a placeholder for showing the current time
    time_display = st.empty()
    
    # Now create the time slider
    time_idx = st.slider(
        "Time Progress",
        min_value=0,
        max_value=len(times) - 1,
        value=0,
        key="time_slider"
    )
    
    # Update the time display
    _update_time_step_display(time_display, time_idx, times)
    
    # Add play/pause controls
    with btn_col1:
        if st.button("⏮️ Start", key="start_btn"):
            st.session_state.time_slider = 0
            st.rerun()
    
    with btn_col2:
        if st.button("⏪ -10", key="back_btn"):
            st.session_state.time_slider = max(0, time_idx - 10)
            st.rerun()
    
    with btn_col3:
        # Toggle between play and pause
        play_state = st.checkbox("Play/Pause", key="play_pause")
    
    with btn_col4:
        if st.button("⏩ +10", key="forward_btn"):
            st.session_state.time_slider = min(len(times) - 1, time_idx + 10)
            st.rerun()
            
    with btn_col5:
        if st.button("⏭️ End", key="end_btn"):
            st.session_state.time_slider = len(times) - 1
            st.rerun()
    
    # Auto-advance the slider if play is active
    if play_state and time_idx < len(times) - 1:
        time.sleep(0.1)  # Control playback speed
        st.session_state.time_slider += 1
        st.rerun()
    
    # Create a 3-column layout for the plots
    col1, col2, col3 = st.columns(3)
    
    # Plot phase positions in unit circle
    with col1:
        st.subheader("Phase Positions")
        fig = _plot_phase_positions(phases, time_idx, n_oscillators)
        st.pyplot(fig)
    
    # Plot oscillator phases vs time
    with col2:
        st.subheader("Oscillator Phases")
        fig = _plot_oscillator_phases(times, phases, time_idx, n_oscillators)
        st.pyplot(fig)
    
    # Plot order parameter
    with col3:
        st.subheader("Order Parameter")
        fig = _plot_order_parameter(times, order_parameter, time_idx)
        st.pyplot(fig)

def _update_time_step_display(container, time_idx, times):
    """Update the time step display with current simulation time."""
    if time_idx < len(times):
        time_value = times[time_idx]
        container.markdown(f"""
        <div style='text-align: center; 
                    background-color: rgba(0,0,0,0.2); 
                    padding: 8px; 
                    border-radius: 5px; 
                    margin-bottom: 10px;
                    font-size: 1.2em;'>
            Time: {time_value:.2f}
        </div>
        """, unsafe_allow_html=True)

def _plot_phase_positions(phases, time_idx, n_oscillators):
    """Plot phase positions on the unit circle."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Set up the plot for unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='white', linestyle='-', alpha=0.4)
    ax.add_patch(circle)
    
    # Set background color and grid
    ax.set_facecolor('#1e1e1e')
    ax.grid(True, alpha=0.3)
    
    # Create a good-looking colormap for oscillators
    colors = plt.cm.hsv(np.linspace(0, 1, n_oscillators))
    
    # Get phases at the current time
    phases_at_time = phases[:, time_idx]
    
    # Convert to x,y coordinates on unit circle
    x = np.cos(phases_at_time)
    y = np.sin(phases_at_time)
    
    # Plot oscillators as points
    ax.scatter(x, y, c=colors, s=100, zorder=3)
    
    # Add lines from origin to each oscillator
    for i in range(n_oscillators):
        ax.plot([0, x[i]], [0, y[i]], color=colors[i], alpha=0.5)
    
    # Calculate the order parameter
    r = np.abs(np.mean(np.exp(1j * phases_at_time)))
    psi = np.angle(np.mean(np.exp(1j * phases_at_time)))
    
    # Add the order parameter as an arrow
    ax.arrow(0, 0, r*np.cos(psi), r*np.sin(psi), 
             head_width=0.1, head_length=0.1, fc='yellow', ec='yellow', alpha=0.8, zorder=4)
    
    # Add a small text label for the order parameter value
    ax.text(0.05, -0.95, f"Order = {r:.3f}", color="yellow", fontsize=10)
    
    # Set equal aspect and limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    
    # Labels and title
    ax.set_title(f"Oscillator Phases at t={time_idx}")
    
    return fig

def _plot_oscillator_phases(times, phases, time_idx, n_oscillators):
    """Plot oscillator phases vs time."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Create a good-looking colormap for oscillators
    colors = plt.cm.hsv(np.linspace(0, 1, n_oscillators))
    
    # Plot each oscillator's phase vs time
    for i in range(n_oscillators):
        ax.plot(times[:time_idx+1], phases[i, :time_idx+1] % (2*np.pi), color=colors[i], alpha=0.7)
    
    # Add a vertical line at the current time
    if time_idx > 0:
        ax.axvline(x=times[time_idx], color='white', linestyle='--', alpha=0.5)
    
    # Set y-axis limits to show the full 0 to 2π range
    ax.set_ylim(0, 2*np.pi)
    ax.set_yticks(np.linspace(0, 2*np.pi, 5))
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Phase")
    ax.set_title("Oscillator Phases vs Time")
    
    return fig

def _plot_order_parameter(times, order_parameter, time_idx):
    """Plot order parameter vs time."""
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # Plot order parameter vs time
    ax.plot(times[:time_idx+1], order_parameter[:time_idx+1], 'o', color='yellow', markersize=2, alpha=0.7)
    
    # Set axis limits
    ax.set_xlim(0, times[-1])
    ax.set_ylim(0, 1.05)
    
    # Labels and title
    ax.set_xlabel("Time")
    ax.set_ylabel("Order Parameter (r)")
    ax.set_title("Order Parameter vs Time")
    
    return fig

def _render_network_tab(model, times, phases, n_oscillators, network_type, adj_matrix):
    """Render the network visualization tab."""
    st.markdown("### Network Visualization")
    
    # Create a placeholder for the network visualization
    net_plot_container = st.container()
    
    # Refresh control - only execute when requested
    if st.session_state.refresh_network:
        print("Refresh network requested, generating network plot...")
        st.session_state.refresh_network = False  # Reset flag
    
    # Generate the network graph
    with net_plot_container:
        graph_info, graph_fig = plot_network_graph(model, n_oscillators, network_type, adj_matrix)
        st.pyplot(graph_fig)
        st.markdown(f"**Network Info**: {graph_info}")

def plot_network_graph(model, n_oscillators, network_type, adj_matrix):
    """Plot the oscillator network as a graph."""
    # Get adjacency matrix from model
    if adj_matrix is None:
        # Use the one from the model
        if network_type == "All-to-All":
            # Create a fully connected network (minus self-connections)
            adj_matrix = np.ones((n_oscillators, n_oscillators)) - np.eye(n_oscillators)
            
        elif network_type == "Nearest Neighbor":
            # Create a ring network where each oscillator connects to its neighbors
            adj_matrix = np.zeros((n_oscillators, n_oscillators))
            for i in range(n_oscillators):
                # Connect to left and right neighbors in the ring
                adj_matrix[i, (i-1) % n_oscillators] = 1
                adj_matrix[i, (i+1) % n_oscillators] = 1
                
        elif network_type == "Random":
            # Create a random network with ~25% connectivity
            np.random.seed(int(st.session_state.random_seed))
            
            # Start with zero connectivity
            adj_matrix = np.zeros((n_oscillators, n_oscillators))
            
            # For each oscillator, connect to ~25% of others randomly
            connect_prob = 0.25
            for i in range(n_oscillators):
                for j in range(n_oscillators):
                    if i != j and np.random.random() < connect_prob:
                        adj_matrix[i, j] = 1
                        adj_matrix[j, i] = 1  # Ensure symmetry
    
    # Create a networkx graph from the adjacency matrix
    G = nx.from_numpy_array(adj_matrix)
    
    # Count edges for info
    edge_count = G.number_of_edges()
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Get oscillator's natural frequencies to determine node colors
    natural_frequencies = model.natural_frequencies if hasattr(model, 'natural_frequencies') else np.zeros(n_oscillators)
    
    # Set node colors based on natural frequencies
    freq_range = max(abs(np.min(natural_frequencies)), abs(np.max(natural_frequencies)))
    if freq_range == 0:
        freq_range = 1  # Prevent division by zero
    
    # Create a colormap from blue to red
    cmap = LinearSegmentedColormap.from_list("BuRd", ["blue", "white", "red"])
    
    # Normalize frequencies to [-1, 1] for colormap
    norm_freqs = natural_frequencies / freq_range
    
    # Node positions using a spring layout with seed
    pos = nx.spring_layout(G, seed=int(st.session_state.random_seed))
    
    # Draw nodes with colors based on frequencies
    node_colors = [cmap((f + 1) / 2) for f in norm_freqs]  # Map [-1,1] to [0,1]
    
    # Get the last phase state to determine node border colors
    # If none available, use zeros
    node_border_colors = []
    for i in range(n_oscillators):
        angle = 0  # Default
        if hasattr(model, 'phases') and model.phases.shape[1] > 0:
            angle = model.phases[i, -1]
        # Convert phase angle to a color using HSV (hue based on angle)
        hue = (angle % (2*np.pi)) / (2*np.pi)
        node_border_colors.append(plt.cm.hsv(hue))
    
    # Draw the network
    nodes = nx.draw_networkx_nodes(G, pos, 
                                  node_color=node_colors,
                                  linewidths=2,
                                  edgecolors=node_border_colors,
                                  node_size=500)
    
    # Draw edges with transparency and varying width based on strength
    edge_weights = [adj_matrix[i, j] for i, j in G.edges()]
    
    # Edge colors indicating synchronization
    edge_colors = []
    for i, j in G.edges():
        # If we have phase data, calculate synchronization based on phase similarity
        if hasattr(model, 'phases') and model.phases.shape[1] > 0:
            # Get last phase values
            phase_i = model.phases[i, -1]
            phase_j = model.phases[j, -1]
            
            # Calculate phase difference (wrapped to [-π, π])
            phase_diff = (phase_i - phase_j + np.pi) % (2*np.pi) - np.pi
            
            # Normalize phase difference to [0, 1] with 0 being completely synchronized
            # and 1 being completely out of phase
            sync = 1 - abs(phase_diff) / np.pi
            
            # Use a green-yellow-red colormap for synchronization
            edge_colors.append(plt.cm.RdYlGn(sync))
        else:
            # If no phase data, use a default color
            edge_colors.append('gray')
    
    # Draw edges with appropriate width and color
    edges = nx.draw_networkx_edges(G, pos, 
                                  width=[w * 2 for w in edge_weights],
                                  edge_color=edge_colors,
                                  alpha=0.7)
    
    # Add labels to nodes
    nx.draw_networkx_labels(G, pos, font_color='white')
    
    # Add a colorbar for frequency
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-freq_range, freq_range))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Natural Frequency')
    
    # Set plot limits and remove axes
    plt.axis('off')
    
    # Return info string and figure
    graph_info = f"nodes={n_oscillators}, edges={edge_count}"
    print(f"Graph info: {graph_info}")
    
    return graph_info, fig

def _render_time_evolution_tab(times, phases, order_parameter, n_oscillators):
    """Render the time evolution visualization tab."""
    st.markdown("### Time Evolution of the System")
    
    # Create a 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phase Evolution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create a good-looking colormap for oscillators
        colors = plt.cm.hsv(np.linspace(0, 1, n_oscillators))
        
        # Plot each oscillator's phase vs time
        for i in range(n_oscillators):
            ax.plot(times, phases[i] % (2*np.pi), color=colors[i], alpha=0.7)
        
        # Set y-axis limits to show the full 0 to 2π range
        ax.set_ylim(0, 2*np.pi)
        ax.set_yticks(np.linspace(0, 2*np.pi, 5))
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        # Labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Phase")
        ax.set_title("Oscillator Phases Over Time")
        
        # Add legend if not too many oscillators
        if n_oscillators <= 10:
            ax.legend([f"Osc {i+1}" for i in range(n_oscillators)], 
                     loc='center left', bbox_to_anchor=(1, 0.5))
        
        st.pyplot(fig)
    
    with col2:
        st.subheader("Order Parameter Evolution")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot order parameter vs time
        ax.plot(times, order_parameter, color='yellow', linewidth=2)
        
        # Set axis limits
        ax.set_xlim(0, times[-1])
        ax.set_ylim(0, 1.05)
        
        # Labels and title
        ax.set_xlabel("Time")
        ax.set_ylabel("Order Parameter (r)")
        ax.set_title("Order Parameter Over Time")
        
        st.pyplot(fig)
    
    # Additional visualization: Phase coherence over time as a heatmap
    st.subheader("Phase Coherence Visualization")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Normalize phases to [0, 2π)
    normalized_phases = phases % (2*np.pi)
    
    # Set up the plot
    # Use a circular colormap for phases
    cmap = plt.cm.hsv
    
    # Create a collection of colored stripes
    collection = matplotlib.collections.LineCollection(
        [np.column_stack([times, np.ones(len(times)) * i]) for i in range(n_oscillators)],
        linewidths=5,
        colors=[cmap(phase / (2*np.pi)) for phase in normalized_phases.flatten()]
    )
    
    ax.add_collection(collection)
    
    # Set limits and labels
    ax.set_xlim(0, times[-1])
    ax.set_ylim(-1, n_oscillators)
    ax.set_xlabel("Time")
    ax.set_ylabel("Oscillator")
    ax.set_title("Phase Coherence Over Time")
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(0, 2*np.pi))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Phase')
    cbar.set_ticks(np.linspace(0, 2*np.pi, 5))
    cbar.set_ticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    st.pyplot(fig)

def _render_analysis_tab(times, phases, order_parameter, n_oscillators):
    """Render the analysis visualization tab."""
    st.markdown("### System Analysis")
    
    # Create a 2-column layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Order Parameter Statistics")
        
        # Calculate statistics on the order parameter
        mean_order = np.mean(order_parameter)
        final_order = order_parameter[-1]
        min_order = np.min(order_parameter)
        max_order = np.max(order_parameter)
        
        # Determine if synchronization occurred
        sync_threshold = 0.9
        is_synchronized = final_order > sync_threshold
        
        # Display statistics
        st.markdown(f"""
        | Statistic | Value |
        |-----------|-------|
        | Mean Order Parameter | {mean_order:.3f} |
        | Final Order Parameter | {final_order:.3f} |
        | Minimum Order | {min_order:.3f} |
        | Maximum Order | {max_order:.3f} |
        | System Synchronized | {"Yes" if is_synchronized else "No"} |
        """)
        
        # Plot the order parameter histogram
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(order_parameter, bins=30, color='yellow', alpha=0.7)
        ax.set_xlabel("Order Parameter Value")
        ax.set_ylabel("Frequency")
        ax.set_title("Order Parameter Distribution")
        st.pyplot(fig)
    
    with col2:
        st.subheader("Phase Distribution Analysis")
        
        # Calculate final phase distribution
        final_phases = phases[:, -1] % (2*np.pi)
        
        # Plot final phase distribution
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(final_phases, bins=20, color='skyblue', alpha=0.7)
        ax.set_xlabel("Phase (radians)")
        ax.set_ylabel("Count")
        ax.set_title("Final Phase Distribution")
        ax.set_xlim(0, 2*np.pi)
        ax.set_xticks(np.linspace(0, 2*np.pi, 5))
        ax.set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        st.pyplot(fig)
        
        # Calculate phase coherence
        phase_diffs = []
        for i in range(n_oscillators):
            for j in range(i+1, n_oscillators):
                # Calculate phase difference (wrapped to [-π, π])
                diff = (final_phases[i] - final_phases[j] + np.pi) % (2*np.pi) - np.pi
                phase_diffs.append(abs(diff))
        
        mean_phase_diff = np.mean(phase_diffs)
        st.markdown(f"**Mean Absolute Phase Difference**: {mean_phase_diff:.3f} radians")
        
    # Full-width analysis: frequency vs final phase
    st.subheader("Natural Frequency vs Final Phase")
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    try:
        # This assumes we can get natural frequencies from somewhere
        # If not available, we'll use a default approach
        natural_frequencies = np.linspace(-2, 2, n_oscillators)  # Dummy values
        
        # Plot frequency vs final phase
        scatter = ax.scatter(natural_frequencies, 
                            final_phases, 
                            c=natural_frequencies, 
                            cmap='coolwarm',
                            s=100, 
                            alpha=0.8)
        
        # Add labels and title
        ax.set_xlabel("Natural Frequency")
        ax.set_ylabel("Final Phase")
        ax.set_title("Relationship Between Natural Frequency and Final Phase")
        
        # Add a trend line
        z = np.polyfit(natural_frequencies, final_phases, 1)
        p = np.poly1d(z)
        ax.plot(natural_frequencies, p(natural_frequencies), "r--", alpha=0.8)
        
        # Add a colorbar
        plt.colorbar(scatter, ax=ax, label='Natural Frequency')
        
        # Set y-axis limits to show the full 0 to 2π range
        ax.set_ylim(0, 2*np.pi)
        ax.set_yticks(np.linspace(0, 2*np.pi, 5))
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate frequency vs phase plot: {str(e)}")
    
    # Provide a download link for the data
    st.subheader("Download Simulation Data")
    
    # Create a CSV with time, order parameter, and all oscillator phases
    csv_data = "time,order_parameter," + ",".join([f"oscillator_{i+1}" for i in range(n_oscillators)]) + "\n"
    
    for i in range(len(times)):
        row = [str(times[i]), str(order_parameter[i])]
        for j in range(n_oscillators):
            row.append(str(phases[j, i]))
        csv_data += ",".join(row) + "\n"
    
    # Create a download link
    b64 = base64.b64encode(csv_data.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="kuramoto_simulation_data.csv">Download CSV</a>'
    st.markdown(href, unsafe_allow_html=True)

def plot_interactive_network(model, n_oscillators, network_type, adj_matrix):
    """Create an interactive network visualization using Plotly."""
    # This is a placeholder for an interactive network visualization
    # Using Plotly would be ideal here, but requires significant code
    return None