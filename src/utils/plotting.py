"""
Plotting utilities for the Kuramoto Model Simulator.
Contains all visualization functions for phase plots, network graphs, and animations.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap


def create_phase_plot(time_idx, times, phases, order_parameter, frequencies, adjacency_matrix=None):
    """Create the unit circle phase visualization plot."""
    # Calculate phases at the specified time index
    phases_at_time = phases[:, time_idx] % (2*np.pi)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a1a')
    
    # Draw unit circle with enhanced styling
    circle_bg = patches.Circle((0, 0), 1, fill=False, edgecolor='#333333', linewidth=2, alpha=0.7)
    ax.add_patch(circle_bg)
    
    # Add subtle inner circles for reference
    for radius in [0.25, 0.5, 0.75]:
        inner_circle = patches.Circle((0, 0), radius, fill=False, edgecolor='#222222', 
                                    linewidth=0.8, alpha=0.3, linestyle='--')
        ax.add_patch(inner_circle)
    
    # Calculate oscillator positions on the circle
    x_positions = np.cos(phases_at_time)
    y_positions = np.sin(phases_at_time)
    
    # Create color map based on natural frequencies
    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    if freq_max != freq_min:
        normalized_frequencies = (frequencies - freq_min) / (freq_max - freq_min)
    else:
        normalized_frequencies = np.ones_like(frequencies) * 0.5
    
    # Use a vibrant color map for oscillators
    colors = plt.cm.plasma(normalized_frequencies)
    
    # Enhanced oscillator dots with glow effect
    for i, (x, y, color) in enumerate(zip(x_positions, y_positions, colors)):
        # Glow effect
        glow_circle = patches.Circle((x, y), 0.08, color=color, alpha=0.3, zorder=1)
        ax.add_patch(glow_circle)
        
        # Main dot
        main_circle = patches.Circle((x, y), 0.05, color=color, alpha=0.9, zorder=2)
        ax.add_patch(main_circle)
    
    # Calculate and draw order parameter vector
    current_order = order_parameter[time_idx]
    mean_phase = np.angle(np.sum(np.exp(1j * phases_at_time)))
    
    # Enhanced arrow for order parameter
    if current_order > 0.01:  # Only draw if significant
        arrow_x = current_order * np.cos(mean_phase)
        arrow_y = current_order * np.sin(mean_phase)
        
        # Glow effect for arrow
        ax.arrow(0, 0, arrow_x, arrow_y, head_width=0.08, head_length=0.06,
                fc='#00e8ff', ec='#00e8ff', alpha=0.3, linewidth=4, zorder=3)
        
        # Main arrow
        ax.arrow(0, 0, arrow_x, arrow_y, head_width=0.06, head_length=0.04,
                fc='#00e8ff', ec='#00e8ff', alpha=0.9, linewidth=2, zorder=4)
    
    # Styling
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal')
    ax.set_title(f'Oscillator Phases at t = {times[time_idx]:.2f}', 
                fontsize=14, fontweight='bold', color='white', pad=15)
    
    # Add phase indicators (0, π/2, π, 3π/2)
    phase_labels = ['0', 'π/2', 'π', '3π/2']
    phase_positions = [(1.15, 0), (0, 1.15), (-1.15, 0), (0, -1.15)]
    for label, (x, y) in zip(phase_labels, phase_positions):
        ax.text(x, y, label, ha='center', va='center', color='white', 
               fontsize=10, fontweight='bold')
    
    # Remove axes and ticks
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    return fig


def create_oscillator_phases_plot(time_idx, times, phases, frequencies):
    """Create the oscillator phases over time plot."""
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a1a')
    
    # Create color map based on natural frequencies
    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    if freq_max != freq_min:
        normalized_frequencies = (frequencies - freq_min) / (freq_max - freq_min)
    else:
        normalized_frequencies = np.ones_like(frequencies) * 0.5
    
    colors = plt.cm.plasma(normalized_frequencies)
    
    # Plot each oscillator's phase trajectory
    for i in range(len(frequencies)):
        phase_trajectory = phases[i, :] % (2*np.pi)
        ax.plot(times, phase_trajectory, color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Add current time indicator
    current_time = times[time_idx]
    ax.axvline(x=current_time, color='#ff5555', linestyle='-', linewidth=2, alpha=0.8)
    
    # Styling
    ax.set_xlabel('Time', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Phase (mod 2π)', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Oscillator Phase Evolution', fontsize=14, fontweight='bold', color='white', pad=15)
    
    # Set y-axis limits and labels
    ax.set_ylim(0, 2*np.pi)
    ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Grid and styling
    ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig


def create_order_parameter_plot(time_idx, times, order_parameter):
    """Create the order parameter over time plot."""
    # Create figure
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1a1a1a')
    
    # Create color map for order parameter values
    colors = plt.cm.viridis(order_parameter)
    
    # Plot order parameter as colored dots
    scatter = ax.scatter(times, order_parameter, c=order_parameter, cmap='viridis', 
                        s=20, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Add current time indicator
    current_time = times[time_idx]
    current_order = order_parameter[time_idx]
    ax.axvline(x=current_time, color='#ff5555', linestyle='-', linewidth=2, alpha=0.8)
    
    # Highlight current point
    ax.scatter([current_time], [current_order], c='#ff5555', s=100, 
              edgecolors='white', linewidth=2, zorder=5)
    
    # Styling
    ax.set_xlabel('Time', fontsize=12, fontweight='bold', color='white')
    ax.set_ylabel('Order Parameter r(t)', fontsize=12, fontweight='bold', color='white')
    ax.set_title('Synchronization Level', fontsize=14, fontweight='bold', color='white', pad=15)
    
    ax.set_ylim(0, 1)
    ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
    ax.tick_params(colors='white')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Synchronization', color='white', fontweight='bold')
    cbar.ax.tick_params(colors='white')
    
    for spine in ax.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1)
    
    plt.tight_layout()
    return fig


def create_network_plot(adjacency_matrix, frequencies, pos=None):
    """Create a network visualization plot."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    
    # Create NetworkX graph
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Generate layout if not provided
    if pos is None:
        if len(G.nodes()) <= 50:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.random_layout(G, seed=42)
    
    # Create color map based on frequencies
    freq_min, freq_max = np.min(frequencies), np.max(frequencies)
    if freq_max != freq_min:
        normalized_frequencies = (frequencies - freq_min) / (freq_max - freq_min)
    else:
        normalized_frequencies = np.ones_like(frequencies) * 0.5
    
    node_colors = plt.cm.plasma(normalized_frequencies)
    
    # Plot 1: Network structure
    ax1.set_facecolor('#1a1a1a')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='#333333', alpha=0.6, width=0.5)
    
    # Draw nodes with frequency-based colors
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                          node_size=100, alpha=0.9)
    
    ax1.set_title('Network Structure', fontsize=14, fontweight='bold', color='white')
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Adjacency matrix heatmap
    ax2.set_facecolor('#1a1a1a')
    
    im = ax2.imshow(adjacency_matrix, cmap='Blues', aspect='equal', interpolation='nearest')
    ax2.set_title('Adjacency Matrix', fontsize=14, fontweight='bold', color='white')
    ax2.set_xlabel('Oscillator Index', fontsize=12, color='white')
    ax2.set_ylabel('Oscillator Index', fontsize=12, color='white')
    ax2.tick_params(colors='white')
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Connection Strength')
    
    plt.tight_layout()
    return fig


def create_frequency_distribution_plot(frequencies, freq_type, freq_params=None):
    """Create frequency distribution visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0e1117')
    
    # Plot 1: Histogram
    ax1.set_facecolor('#1a1a1a')
    
    n_bins = min(20, len(frequencies) // 2) if len(frequencies) > 4 else len(frequencies)
    counts, bin_edges, patches = ax1.hist(frequencies, bins=n_bins, alpha=0.8, 
                                         edgecolor='white', linewidth=0.5)
    
    # Color bars based on frequency values
    colors = plt.cm.plasma((bin_edges[:-1] - np.min(frequencies)) / 
                          (np.max(frequencies) - np.min(frequencies)) if np.max(frequencies) != np.min(frequencies) else 0.5)
    
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    
    ax1.set_xlabel('Natural Frequency', fontsize=12, fontweight='bold', color='white')
    ax1.set_ylabel('Count', fontsize=12, fontweight='bold', color='white')
    ax1.set_title(f'{freq_type} Distribution', fontsize=14, fontweight='bold', color='white')
    ax1.tick_params(colors='white')
    ax1.grid(True, color='#333333', alpha=0.4)
    
    # Plot 2: Oscillator frequency mapping
    ax2.set_facecolor('#1a1a1a')
    
    oscillator_indices = np.arange(len(frequencies))
    colors = plt.cm.plasma((frequencies - np.min(frequencies)) / 
                          (np.max(frequencies) - np.min(frequencies)) if np.max(frequencies) != np.min(frequencies) else 0.5)
    
    scatter = ax2.scatter(oscillator_indices, frequencies, c=frequencies, 
                         cmap='plasma', s=60, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    ax2.set_xlabel('Oscillator Index', fontsize=12, fontweight='bold', color='white')
    ax2.set_ylabel('Natural Frequency', fontsize=12, fontweight='bold', color='white')
    ax2.set_title('Frequency Assignment', fontsize=14, fontweight='bold', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, color='#333333', alpha=0.4)
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax2, label='Frequency Value')
    
    plt.tight_layout()
    return fig