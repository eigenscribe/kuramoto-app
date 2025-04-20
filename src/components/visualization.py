"""
Visualization components for the Kuramoto simulator.
This module contains functions to create visualizations for the simulation.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import matplotlib.collections as mcoll
import plotly.graph_objects as go
import networkx as nx
from io import BytesIO
import base64

def create_circular_layout(G, radius=1.0):
    """
    Create a circular layout for a network.
    
    Parameters:
    -----------
    G : NetworkX graph object
        The graph to lay out
    radius : float
        Radius of the circle
        
    Returns:
    --------
    dict
        Dictionary of node positions {node: (x, y)}
    """
    pos = {}
    n_nodes = len(G.nodes())
    
    for i, node in enumerate(G.nodes()):
        theta = 2.0 * np.pi * i / n_nodes
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        pos[node] = (x, y)
        
    return pos

def plot_network(adjacency_matrix, node_colors=None, edge_threshold=0, title=None, ax=None, figsize=(10, 10)):
    """
    Visualize the network structure.
    
    Parameters:
    -----------
    adjacency_matrix : ndarray
        The adjacency matrix of the network
    node_colors : list, optional
        Colors for each node
    edge_threshold : float, optional
        Only show edges with weight above this threshold
    title : str, optional
        Title for the plot
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Remove edges below threshold
    if edge_threshold > 0:
        edges_to_remove = [(i, j) for i, j, w in G.edges(data='weight') if w < edge_threshold]
        G.remove_edges_from(edges_to_remove)
    
    # Get positions in a circle
    pos = create_circular_layout(G)
    
    # Set default node colors if not provided
    if node_colors is None:
        node_colors = ['#1f77b4'] * len(G.nodes())
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_color=node_colors,
        node_size=800, 
        alpha=0.9,
        edgecolors='white',
        linewidths=2,
        ax=ax
    )
    
    # Create a colormap for edges based on weight
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    if edge_weights:
        min_weight = min(edge_weights)
        max_weight = max(edge_weights)
        norm = mcolors.Normalize(vmin=min_weight, vmax=max_weight)
        cmap = plt.cm.Blues
        
        # Draw edges with varying colors and widths
        for u, v, weight in G.edges(data='weight'):
            edge_color = cmap(norm(weight))
            edge_width = 1 + 5 * norm(weight)
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=edge_width,
                alpha=0.7,
                edge_color=[edge_color],
                ax=ax
            )
    
    # Add a colorbar if there are edges
    if edge_weights:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Edge Weight')
    
    # Set the title
    if title:
        ax.set_title(title, fontsize=16, fontweight='bold')
    
    # Remove axis
    ax.set_axis_off()
    
    # Add a circle to represent the unit circle
    circle = plt.Circle((0, 0), radius=1.0, fill=False, color='black', linestyle='--', linewidth=1)
    ax.add_patch(circle)
    
    # Set limits slightly larger than the circle
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig

def plot_oscillator_phases(phases, times, time_idx=0, ax=None, figsize=(10, 6)):
    """
    Plot the phases of all oscillators at a specific time.
    
    Parameters:
    -----------
    phases : ndarray
        Phases of oscillators over time (shape: n_oscillators x n_times)
    times : ndarray
        Time points
    time_idx : int, optional
        Index of time point to plot
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Get phases at the specified time
    phases_at_time = phases[:, time_idx]
    n_oscillators = phases.shape[0]
    
    # Create color map based on natural frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
    
    # Plot phases on a circle
    for i, phase in enumerate(phases_at_time):
        x = np.cos(phase)
        y = np.sin(phase)
        ax.plot([0, x], [0, y], color=colors[i], alpha=0.7, lw=2)
        ax.scatter(x, y, s=100, color=colors[i], edgecolor='white', linewidth=1, zorder=10)
    
    # Calculate order parameter
    r = np.abs(np.sum(np.exp(1j * phases_at_time))) / n_oscillators
    psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
    
    # Plot mean field vector
    ax.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
             head_width=0.05, head_length=0.1, fc='red', ec='red', 
             width=0.02, zorder=5)
    
    # Add a circle to represent the unit circle
    circle = plt.Circle((0, 0), radius=1.0, fill=False, color='black', linestyle='--', linewidth=1)
    ax.add_patch(circle)
    
    # Set limits
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    
    # Add title and info
    ax.set_title(f'Oscillator Phases at t={times[time_idx]:.2f} (Order Parameter: r={r:.3f})', fontsize=14)
    
    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Add grid
    ax.grid(linestyle=':', alpha=0.3)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal')
    
    # Add annotation for order parameter
    ax.text(0.02, 0.02, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, 
            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Add colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, n_oscillators-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Oscillator Index')
    
    plt.tight_layout()
    
    return fig

def plot_order_parameter(times, order_parameter, ax=None, figsize=(10, 6)):
    """
    Plot the order parameter over time.
    
    Parameters:
    -----------
    times : ndarray
        Time points
    order_parameter : ndarray
        Order parameter values over time
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot order parameter
    ax.plot(times, order_parameter, color='#1f77b4', linewidth=2)
    
    # Fill area under the curve
    ax.fill_between(times, 0, order_parameter, color='#1f77b4', alpha=0.3)
    
    # Add labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Order Parameter (r)', fontsize=12)
    ax.set_title('Kuramoto Model Synchronization', fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0, 1.05)
    
    # Add grid
    ax.grid(linestyle=':', alpha=0.3)
    
    # Annotate final value
    final_r = order_parameter[-1]
    ax.annotate(
        f'Final r = {final_r:.3f}',
        xy=(times[-1], final_r),
        xytext=(times[-1] - 0.1 * (times[-1] - times[0]), final_r + 0.1),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
        bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
    )
    
    plt.tight_layout()
    
    return fig

def plot_phase_evolution(phases, times, ax=None, figsize=(10, 6)):
    """
    Plot the phase evolution of all oscillators over time.
    
    Parameters:
    -----------
    phases : ndarray
        Phases of oscillators over time (shape: n_oscillators x n_times)
    times : ndarray
        Time points
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    n_oscillators = phases.shape[0]
    
    # Create color map based on natural frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
    
    # Plot phases
    for i in range(n_oscillators):
        ax.plot(times, phases[i, :] % (2 * np.pi), color=colors[i], alpha=0.7, lw=1)
    
    # Add labels and title
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Phase (mod 2π)', fontsize=12)
    ax.set_title('Oscillator Phases Over Time', fontsize=14)
    
    # Set y-axis limits
    ax.set_ylim(0, 2 * np.pi)
    
    # Add y-ticks at π intervals
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Add grid
    ax.grid(linestyle=':', alpha=0.3)
    
    # Add colorbar for reference
    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, n_oscillators-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Oscillator Index')
    
    plt.tight_layout()
    
    return fig

def plot_frequency_histogram(frequencies, freq_distribution, freq_params=None, ax=None, figsize=(10, 6)):
    """
    Plot histogram of natural frequencies.
    
    Parameters:
    -----------
    frequencies : ndarray
        Natural frequencies of oscillators
    freq_distribution : str
        Type of frequency distribution
    freq_params : dict, optional
        Parameters of the frequency distribution
    ax : matplotlib.axes.Axes, optional
        Axes on which to plot
    figsize : tuple, optional
        Figure size (width, height) in inches
        
    Returns:
    --------
    fig : matplotlib.figure.Figure
        The figure containing the plot
    """
    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    
    # Plot histogram
    bins = min(20, len(frequencies))
    n, bins, patches = ax.hist(frequencies, bins=bins, alpha=0.7, color='#1f77b4', density=True)
    
    # Calculate mean and standard deviation
    mean = np.mean(frequencies)
    std = np.std(frequencies)
    
    # Get x range for plotting theoretical distributions
    x = np.linspace(min(frequencies) - 1, max(frequencies) + 1, 1000)
    
    # Plot theoretical distribution if parameters are available
    if freq_params:
        if freq_distribution == "Normal":
            # Normal distribution
            pdf = (1 / (freq_params['std'] * np.sqrt(2 * np.pi))) * np.exp(-(x - freq_params['mean'])**2 / (2 * freq_params['std']**2))
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal(μ={freq_params["mean"]:.2f}, σ={freq_params["std"]:.2f})')
        elif freq_distribution == "Uniform":
            # Uniform distribution
            pdf = np.zeros_like(x)
            mask = (x >= freq_params['min']) & (x <= freq_params['max'])
            pdf[mask] = 1.0 / (freq_params['max'] - freq_params['min'])
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Uniform({freq_params["min"]:.2f}, {freq_params["max"]:.2f})')
        elif freq_distribution == "Bimodal":
            # Bimodal distribution (mixture of two normals)
            std = 0.3  # Fixed std for bimodal
            pdf1 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - freq_params['peak1'])**2 / (2 * std**2))
            pdf2 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - freq_params['peak2'])**2 / (2 * std**2))
            pdf = 0.5 * (pdf1 + pdf2)
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Bimodal({freq_params["peak1"]:.2f}, {freq_params["peak2"]:.2f})')
    
    # Add labels and title
    ax.set_xlabel('Natural Frequency (ω)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Natural Frequency Distribution ({freq_distribution})', fontsize=14)
    
    # Add text with summary statistics
    stats_text = f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\nMin: {min(frequencies):.3f}\nMax: {max(frequencies):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add legend if theoretical distribution is plotted
    if freq_params:
        ax.legend()
    
    # Add grid
    ax.grid(linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    return fig

def create_animation_html(phases, times, save_path=None):
    """
    Create an animation of oscillator phases.
    
    Parameters:
    -----------
    phases : ndarray
        Phases of oscillators over time (shape: n_oscillators x n_times)
    times : ndarray
        Time points
    save_path : str, optional
        Path to save the animation HTML
        
    Returns:
    --------
    str
        HTML code for the animation
    """
    n_oscillators = phases.shape[0]
    n_times = phases.shape[1]
    
    # Create figure and axes
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Set up the figure
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.grid(linestyle=':', alpha=0.3)
    ax.set_title('Oscillator Phases Animation', fontsize=14)
    
    # Add a circle to represent the unit circle
    circle = plt.Circle((0, 0), radius=1.0, fill=False, color='black', linestyle='--', linewidth=1)
    ax.add_patch(circle)
    
    # Create color map based on natural frequencies
    colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
    
    # Initialize oscillator points and lines
    points = ax.scatter([], [], s=100, edgecolor='white', linewidth=1, zorder=10)
    lines = [ax.plot([], [], color=colors[i], alpha=0.7, lw=2)[0] for i in range(n_oscillators)]
    
    # Initialize order parameter arrow
    arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, fc='red', ec='red', width=0.02, zorder=5)
    arrow_patch = arrow.get_patch_transform()
    
    # Text for order parameter value
    r_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12, 
                     bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Animation function
    def update(frame):
        # Get phases at the current time
        current_phases = phases[:, frame]
        
        # Update oscillator positions
        x = np.cos(current_phases)
        y = np.sin(current_phases)
        
        # Update scatter points
        points.set_offsets(np.column_stack([x, y]))
        points.set_color(colors)
        
        # Update lines
        for i, line in enumerate(lines):
            line.set_data([0, x[i]], [0, y[i]])
        
        # Calculate order parameter
        r = np.abs(np.sum(np.exp(1j * current_phases))) / n_oscillators
        psi = np.angle(np.sum(np.exp(1j * current_phases)))
        
        # Update order parameter arrow
        arrow.set_data(x=0, y=0, dx=r * np.cos(psi), dy=r * np.sin(psi))
        
        # Update text
        r_text.set_text(f'r = {r:.3f}')
        
        # Update title
        ax.set_title(f'Oscillator Phases at t={times[frame]:.2f}', fontsize=14)
        
        return [points, *lines, arrow, r_text]
    
    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=min(100, n_times), interval=50, blit=True)
    
    # Save as HTML if save_path is specified
    if save_path:
        anim.save(save_path, writer='pillow')
        
    # Convert to HTML
    html = anim.to_jshtml()
    
    plt.close(fig)
    
    return html

def fig_to_base64(fig):
    """
    Convert a matplotlib figure to base64 string for embedding in HTML.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure to convert
        
    Returns:
    --------
    str
        Base64 encoded string of the figure
    """
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str