"""
Network component for the Kuramoto simulator.
This module provides functions for generating and visualizing network structures.
"""

import streamlit as st
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import io
import base64

def generate_network(network_type, n_oscillators, **kwargs):
    """
    Generate a network structure based on the specified type.
    
    Parameters:
    -----------
    network_type : str
        Type of network ("All-to-All", "Random", "Small-World", "Scale-Free", "Lattice", "Custom")
    n_oscillators : int
        Number of oscillators (nodes) in the network
    **kwargs : dict
        Additional parameters for specific network types
    
    Returns:
    --------
    ndarray
        Adjacency matrix of the network
    """
    if network_type == "All-to-All":
        # Fully connected network
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(adj_matrix, 0)  # Remove self-connections
        
    elif network_type == "Random":
        # Erdős–Rényi random graph
        connection_prob = kwargs.get('connection_prob', 0.3)
        G = nx.erdos_renyi_graph(n_oscillators, connection_prob)
        adj_matrix = nx.to_numpy_array(G)
        
    elif network_type == "Small-World":
        # Watts-Strogatz small-world graph
        k_neighbors = kwargs.get('k_neighbors', 4)
        rewire_prob = kwargs.get('rewire_prob', 0.1)
        
        # Ensure k is even and less than n
        k = min(k_neighbors, n_oscillators - 1)
        k = k if k % 2 == 0 else k - 1
        
        G = nx.watts_strogatz_graph(n_oscillators, k, rewire_prob)
        adj_matrix = nx.to_numpy_array(G)
        
    elif network_type == "Scale-Free":
        # Barabási–Albert scale-free graph
        sf_m = kwargs.get('sf_m', 2)
        
        # Ensure m is valid
        m = min(sf_m, n_oscillators - 1)
        
        G = nx.barabasi_albert_graph(n_oscillators, m)
        adj_matrix = nx.to_numpy_array(G)
        
    elif network_type == "Lattice":
        # Regular lattice
        dimensions = kwargs.get('dimensions', 2)
        
        if dimensions == 1:
            # 1D lattice with periodic boundary conditions
            G = nx.cycle_graph(n_oscillators)
        else:
            # 2D lattice
            # Find grid dimensions to approximate a square
            side_length = int(np.ceil(np.sqrt(n_oscillators)))
            
            # Create a 2D grid graph
            G = nx.grid_2d_graph(side_length, (n_oscillators // side_length) + (1 if n_oscillators % side_length else 0))
            
            # Relabel nodes to be integers
            G = nx.convert_node_labels_to_integers(G)
            
            # Remove excess nodes if any
            if G.number_of_nodes() > n_oscillators:
                nodes_to_remove = list(range(n_oscillators, G.number_of_nodes()))
                G.remove_nodes_from(nodes_to_remove)
        
        adj_matrix = nx.to_numpy_array(G)
        
    elif network_type == "Custom":
        # Custom adjacency matrix (provided externally)
        adj_matrix = kwargs.get('adjacency_matrix', None)
        
        if adj_matrix is None:
            # Default to fully connected if no custom matrix provided
            adj_matrix = np.ones((n_oscillators, n_oscillators))
            np.fill_diagonal(adj_matrix, 0)
    else:
        # Default to fully connected
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(adj_matrix, 0)
    
    return adj_matrix

def plot_network(adjacency_matrix, node_size=300, edge_alpha=0.5, colormap='viridis'):
    """
    Visualize a network from its adjacency matrix.
    
    Parameters:
    -----------
    adjacency_matrix : ndarray
        Adjacency matrix of the network
    node_size : int, optional
        Size of nodes in the visualization
    edge_alpha : float, optional
        Transparency of edges
    colormap : str, optional
        Colormap for node colors
    
    Returns:
    --------
    str
        Base64 encoded PNG image of the network visualization
    """
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Calculate node metrics for coloring
    degree_centrality = nx.degree_centrality(G)
    node_colors = [degree_centrality[i] for i in range(len(G))]
    
    # Create a circular layout
    pos = nx.circular_layout(G)
    
    # Draw the graph
    nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_colors, 
                          cmap=plt.cm.get_cmap(colormap), edgecolors='white')
    
    # Draw edges with varying width based on weight
    for (u, v, w) in G.edges(data='weight'):
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=1 + 2 * w, alpha=edge_alpha)
    
    # Remove axis
    ax.set_axis_off()
    
    # Create a colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap(colormap))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Node Degree Centrality')
    
    # Set title
    ax.set_title(f'Network Structure ({len(G)} nodes)', fontsize=14)
    
    # Convert figure to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    return img_str

def render_network_tab(network_params):
    """
    Render the network tab with network visualization and custom matrix options.
    
    Parameters:
    -----------
    network_params : dict
        Parameters for network generation
    
    Returns:
    --------
    ndarray
        Adjacency matrix to use for simulation
    """
    st.markdown("## 🔗 Network Connectivity")
    
    network_type = network_params.get('network_type', 'All-to-All')
    n_oscillators = network_params.get('n_oscillators', 10)
    
    st.markdown(f"### Network Type: {network_type}")
    
    # Initialize adjacency matrix based on network type
    adjacency_matrix = None
    
    # Handle custom adjacency matrix upload
    if network_type == "Custom":
        st.markdown("#### Upload Custom Adjacency Matrix")
        st.markdown("Upload a CSV file with the adjacency matrix or enter values manually.")
        
        upload_method = st.radio(
            "Choose input method:",
            options=["Upload CSV", "Manual Entry"]
        )
        
        if upload_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload adjacency matrix CSV", type=['csv', 'txt'])
            
            if uploaded_file is not None:
                try:
                    # Load matrix from file
                    import pandas as pd
                    matrix_df = pd.read_csv(uploaded_file, header=None)
                    custom_adj_matrix = matrix_df.values
                    
                    # Validate matrix dimensions
                    if custom_adj_matrix.shape != (n_oscillators, n_oscillators):
                        st.warning(f"Matrix dimensions ({custom_adj_matrix.shape[0]}x{custom_adj_matrix.shape[1]}) don't match the number of oscillators ({n_oscillators}). Matrix will be resized.")
                        
                        # Resize matrix if needed
                        if custom_adj_matrix.shape[0] > n_oscillators or custom_adj_matrix.shape[1] > n_oscillators:
                            # Truncate if larger
                            custom_adj_matrix = custom_adj_matrix[:n_oscillators, :n_oscillators]
                        else:
                            # Pad with zeros if smaller
                            padded_matrix = np.zeros((n_oscillators, n_oscillators))
                            padded_matrix[:custom_adj_matrix.shape[0], :custom_adj_matrix.shape[1]] = custom_adj_matrix
                            custom_adj_matrix = padded_matrix
                    
                    st.success("Custom adjacency matrix loaded successfully.")
                    adjacency_matrix = custom_adj_matrix
                    
                    # Display a preview
                    st.markdown("#### Matrix Preview:")
                    st.dataframe(custom_adj_matrix)
                except Exception as e:
                    st.error(f"Error loading matrix: {str(e)}")
        
        elif upload_method == "Manual Entry":
            st.markdown(f"Enter a {n_oscillators}x{n_oscillators} adjacency matrix:")
            
            # Create text area for manual input
            matrix_text = st.text_area(
                "Enter comma-separated values, one row per line:",
                height=150,
                help="Example: 0,1,1,0\\n1,0,0,1\\n1,0,0,1\\n0,1,1,0"
            )
            
            if matrix_text:
                try:
                    # Parse matrix from text
                    rows = matrix_text.strip().split('\n')
                    custom_adj_matrix = []
                    
                    for row in rows:
                        values = [float(val.strip()) for val in row.split(',')]
                        custom_adj_matrix.append(values)
                    
                    custom_adj_matrix = np.array(custom_adj_matrix)
                    
                    # Validate matrix dimensions
                    if custom_adj_matrix.shape != (n_oscillators, n_oscillators):
                        st.warning(f"Matrix dimensions ({custom_adj_matrix.shape[0]}x{custom_adj_matrix.shape[1]}) don't match the number of oscillators ({n_oscillators}). Matrix will be resized.")
                        
                        # Resize matrix if needed
                        if custom_adj_matrix.shape[0] > n_oscillators or custom_adj_matrix.shape[1] > n_oscillators:
                            # Truncate if larger
                            custom_adj_matrix = custom_adj_matrix[:n_oscillators, :n_oscillators]
                        else:
                            # Pad with zeros if smaller
                            padded_matrix = np.zeros((n_oscillators, n_oscillators))
                            padded_matrix[:custom_adj_matrix.shape[0], :custom_adj_matrix.shape[1]] = custom_adj_matrix
                            custom_adj_matrix = padded_matrix
                    
                    st.success("Custom adjacency matrix created successfully.")
                    adjacency_matrix = custom_adj_matrix
                    
                    # Display a preview
                    st.markdown("#### Matrix Preview:")
                    st.dataframe(custom_adj_matrix)
                except Exception as e:
                    st.error(f"Error parsing matrix: {str(e)}")
        
        # Generate a default matrix if none is provided
        if adjacency_matrix is None:
            st.markdown("#### Default Network")
            st.markdown("Using a fully connected network as default.")
            adjacency_matrix = generate_network("All-to-All", n_oscillators)
    
    else:
        # Generate network based on selected type and parameters
        network_kwargs = {k: v for k, v in network_params.items() if k not in ['network_type', 'n_oscillators']}
        adjacency_matrix = generate_network(network_type, n_oscillators, **network_kwargs)
    
    # Visualize the network
    st.markdown("### Network Visualization")
    img_str = plot_network(adjacency_matrix)
    st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
    
    # Display network statistics
    st.markdown("### Network Statistics")
    
    # Create NetworkX graph for calculations
    G = nx.from_numpy_array(adjacency_matrix)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Density
        density = nx.density(G)
        st.metric("Network Density", f"{density:.3f}")
        
        # Average clustering
        avg_clustering = nx.average_clustering(G)
        st.metric("Avg. Clustering", f"{avg_clustering:.3f}")
    
    with col2:
        # Average degree
        avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
        st.metric("Avg. Degree", f"{avg_degree:.2f}")
        
        # Diameter (only for connected graphs)
        if nx.is_connected(G):
            diameter = nx.diameter(G)
            st.metric("Diameter", diameter)
        else:
            st.metric("Diameter", "N/A (Disconnected)")
    
    with col3:
        # Number of edges
        num_edges = G.number_of_edges()
        st.metric("Number of Edges", num_edges)
        
        # Number of connected components
        num_components = nx.number_connected_components(G)
        st.metric("Connected Components", num_components)
    
    # Option to save adjacency matrix
    save_adj_matrix = st.checkbox("Use this network for simulation", value=True)
    
    if save_adj_matrix:
        return adjacency_matrix
    else:
        return None