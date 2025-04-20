import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
from kuramoto_model import KuramotoModel
import time
from database import (store_simulation, get_simulation, list_simulations, delete_simulation,
                  save_configuration, list_configurations, get_configuration, delete_configuration,
                  get_configuration_by_name)

# Initialize session state for configuration loading
if 'loaded_config' not in st.session_state:
    st.session_state.loaded_config = None

# Apply loaded configuration if available
if st.session_state.loaded_config is not None:
    config = st.session_state.loaded_config
    
    # Update session state with configuration values
    st.session_state.n_oscillators = config['n_oscillators']
    st.session_state.coupling_strength = config['coupling_strength']
    st.session_state.simulation_time = config['simulation_time']
    st.session_state.time_step = config['time_step']
    st.session_state.random_seed = config['random_seed']
    st.session_state.network_type = config['network_type']
    st.session_state.freq_type = config['frequency_distribution']
    
    # Update frequency distribution parameters based on type
    freq_params = config.get('frequency_params', {})
    if freq_params:
        try:
            freq_params = json.loads(freq_params)
        except:
            # It's already a dictionary
            pass
            
        if config['frequency_distribution'] == "Normal":
            st.session_state.freq_mean = freq_params.get('mean', 0.0)
            st.session_state.freq_std = freq_params.get('std', 1.0)
        elif config['frequency_distribution'] == "Uniform":
            st.session_state.freq_min = freq_params.get('min', -1.0)
            st.session_state.freq_max = freq_params.get('max', 1.0)
        elif config['frequency_distribution'] == "Bimodal":
            st.session_state.peak1 = freq_params.get('peak1', -1.0)
            st.session_state.peak2 = freq_params.get('peak2', 1.0)
        elif config['frequency_distribution'] == "Custom":
            if 'custom_values' in freq_params and isinstance(freq_params['custom_values'], list):
                st.session_state.custom_freqs = ', '.join(map(str, freq_params['custom_values']))
            elif 'values' in freq_params and isinstance(freq_params['values'], list):
                st.session_state.custom_freqs = ', '.join(map(str, freq_params['values']))
    
    # Handle custom adjacency matrix if present
    if config['network_type'] == "Custom Adjacency Matrix" and config.get('adjacency_matrix') is not None:
        try:
            matrix = config['adjacency_matrix']
            if isinstance(matrix, bytes):
                import pickle
                matrix = pickle.loads(matrix)
                
            # Convert matrix to string representation for the text area
            matrix_str = ""
            for row in matrix:
                matrix_str += ", ".join(str(val) for val in row) + "\n"
            st.session_state.adj_matrix_input = matrix_str.strip()
        except Exception as e:
            st.warning(f"Could not load custom adjacency matrix: {str(e)}")
    
    # Clear the loaded config to prevent reapplying it on next rerun
    st.session_state.loaded_config = None

# Set up Matplotlib style for dark theme plots
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#1e1e1e',
    'figure.facecolor': '#1e1e1e',
    'savefig.facecolor': '#1e1e1e',
    'axes.grid': True,
    'grid.color': '#444444',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.edgecolor': '#444444',
    'xtick.color': '#888888',
    'ytick.color': '#888888',
    'text.color': '#ffffff',
    'axes.labelcolor': '#ffffff',
    'axes.titlecolor': '#ffffff',
    'lines.linewidth': 2,
})

# Set page title and configuration
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import Aclonica font from Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Aclonica&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Get the base64 encoded image
import base64
with open("wisp.base64", "r") as f:
    encoded_image = f.read()

# Add custom background and custom font
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Aclonica&display=swap');
    
    .stApp {{
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                         url('data:image/jpeg;base64,{encoded_image}');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    
    /* Ensure Aclonica font is applied everywhere */
    body, div, p, h1, h2, h3, h4, h5, h6, li, span, label, button, .sidebar .sidebar-content {{
        font-family: 'Aclonica', sans-serif !important;
    }}
    
    /* Fix Streamlit buttons to use Aclonica */
    button, .stButton button, .stDownloadButton button {{
        font-family: 'Aclonica', sans-serif !important;
    }}
    
    /* Fix Streamlit widgets text */
    .stSlider label, .stSelectbox label, .stNumberInput label {{
        font-family: 'Aclonica', sans-serif !important;
    }}
    
    /* Apply gradient_text1 to sidebar labels */
    .sidebar .sidebar-content label {{
        background: -webkit-linear-gradient(#14a5ff, #8138ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)

# Create sidebar with parameters
st.sidebar.markdown("<h2 class='gradient_text1'>Simulation Parameters</h2>", unsafe_allow_html=True)

# Initialize session state for parameters if they don't exist
if 'n_oscillators' not in st.session_state:
    st.session_state.n_oscillators = 10
if 'coupling_strength' not in st.session_state:
    st.session_state.coupling_strength = 1.0
if 'freq_type' not in st.session_state:
    st.session_state.freq_type = "Normal"
if 'freq_mean' not in st.session_state:
    st.session_state.freq_mean = 0.0
if 'freq_std' not in st.session_state:
    st.session_state.freq_std = 1.0
if 'freq_min' not in st.session_state:
    st.session_state.freq_min = -1.0
if 'freq_max' not in st.session_state:
    st.session_state.freq_max = 1.0
if 'peak1' not in st.session_state:
    st.session_state.peak1 = -1.0
if 'peak2' not in st.session_state:
    st.session_state.peak2 = 1.0
if 'custom_freqs' not in st.session_state:
    st.session_state.custom_freqs = "0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -1.5, -2.0"
if 'simulation_time' not in st.session_state:
    st.session_state.simulation_time = 20.0
if 'time_step' not in st.session_state:
    st.session_state.time_step = 0.05
if 'random_seed' not in st.session_state:
    st.session_state.random_seed = 42
if 'network_type' not in st.session_state:
    st.session_state.network_type = "All-to-All"
if 'adj_matrix_input' not in st.session_state:
    st.session_state.adj_matrix_input = ""

# Number of oscillators slider
n_oscillators = st.sidebar.slider(
    "Number of Oscillators",
    min_value=2,
    max_value=50,
    value=st.session_state.n_oscillators,
    step=1,
    help="Number of oscillators in the system",
    key="n_oscillators"
)

# Coupling strength slider
coupling_strength = st.sidebar.slider(
    "Coupling Strength (K)",
    min_value=0.0,
    max_value=10.0,
    value=st.session_state.coupling_strength,
    step=0.1,
    help="Strength of coupling between oscillators",
    key="coupling_strength"
)

# Frequency distribution type
freq_type = st.sidebar.selectbox(
    "Frequency Distribution",
    ["Normal", "Uniform", "Bimodal", "Custom"],
    index=["Normal", "Uniform", "Bimodal", "Custom"].index(st.session_state.freq_type),
    help="Distribution of natural frequencies",
    key="freq_type"
)

# Parameters for frequency distribution
if freq_type == "Normal":
    freq_mean = st.sidebar.slider("Mean", -2.0, 2.0, st.session_state.freq_mean, 0.1, key="freq_mean")
    freq_std = st.sidebar.slider("Standard Deviation", 0.1, 3.0, st.session_state.freq_std, 0.1, key="freq_std")
    frequencies = np.random.normal(freq_mean, freq_std, n_oscillators)
    
elif freq_type == "Uniform":
    freq_min = st.sidebar.slider("Minimum", -5.0, 0.0, st.session_state.freq_min, 0.1, key="freq_min")
    freq_max = st.sidebar.slider("Maximum", 0.0, 5.0, st.session_state.freq_max, 0.1, key="freq_max")
    frequencies = np.random.uniform(freq_min, freq_max, n_oscillators)
    
elif freq_type == "Bimodal":
    peak1 = st.sidebar.slider("Peak 1", -5.0, 0.0, st.session_state.peak1, 0.1, key="peak1")
    peak2 = st.sidebar.slider("Peak 2", 0.0, 5.0, st.session_state.peak2, 0.1, key="peak2")
    mix = np.random.choice([0, 1], size=n_oscillators)
    freq1 = np.random.normal(peak1, 0.3, n_oscillators)
    freq2 = np.random.normal(peak2, 0.3, n_oscillators)
    frequencies = mix * freq1 + (1 - mix) * freq2
    
else:  # Custom
    custom_freqs = st.sidebar.text_area(
        "Enter custom frequencies (comma-separated)",
        value=st.session_state.custom_freqs,
        key="custom_freqs"
    )
    try:
        frequencies = np.array([float(x.strip()) for x in custom_freqs.split(',')])
        # Ensure we have the right number of frequencies
        if len(frequencies) < n_oscillators:
            # Repeat the pattern if not enough values
            frequencies = np.tile(frequencies, int(np.ceil(n_oscillators / len(frequencies))))
        frequencies = frequencies[:n_oscillators]  # Trim if too many
    except:
        st.sidebar.error("Invalid frequency input. Using normal distribution instead.")
        frequencies = np.random.normal(0, 1, n_oscillators)

# Simulation time parameters
simulation_time = st.sidebar.slider(
    "Simulation Time",
    min_value=1.0,
    max_value=100.0,
    value=st.session_state.simulation_time,
    step=1.0,
    help="Total simulation time",
    key="simulation_time"
)

time_step = st.sidebar.slider(
    "Time Step",
    min_value=0.01,
    max_value=0.1,
    value=st.session_state.time_step,
    step=0.01,
    help="Time step for simulation",
    key="time_step"
)

# Initialize model with specified parameters
random_seed = st.sidebar.number_input(
    "Random Seed", 
    value=st.session_state.random_seed, 
    help="Seed for reproducibility",
    key="random_seed"
)

# Network Connectivity Configuration
st.sidebar.markdown("<h3 class='gradient_text1'>Network Connectivity</h3>", unsafe_allow_html=True)
network_type = st.sidebar.radio(
    "Network Type",
    options=["All-to-All", "Nearest Neighbor", "Random", "Custom Adjacency Matrix"],
    index=["All-to-All", "Nearest Neighbor", "Random", "Custom Adjacency Matrix"].index(st.session_state.network_type),
    help="Define how oscillators are connected to each other",
    key="network_type"
)

# Custom adjacency matrix input
adj_matrix = None
if network_type == "Custom Adjacency Matrix":
    st.sidebar.markdown("""
    <div style="font-size: 0.85em;">
    Enter your adjacency matrix as comma-separated values. Each row should be on a new line.
    <br>Example for 3 oscillators:
    <pre style="background-color: #222; padding: 5px; border-radius: 3px;">
0, 1, 0.5
1, 0, 0.8
0.5, 0.8, 0</pre>
    </div>
    """, unsafe_allow_html=True)
    
    adj_matrix_input = st.sidebar.text_area(
        "Adjacency Matrix",
        value=st.session_state.adj_matrix_input,
        height=150,
        help="Enter the adjacency matrix as comma-separated values, each row on a new line",
        key="adj_matrix_input"
    )
    
    # Process the input adjacency matrix
    if adj_matrix_input:
        try:
            # Parse the input text into a numpy array
            rows = adj_matrix_input.strip().split('\n')
            adj_matrix = []
            for row in rows:
                values = [float(val.strip()) for val in row.split(',')]
                adj_matrix.append(values)
            
            adj_matrix = np.array(adj_matrix)
            
            # Validate the adjacency matrix
            if adj_matrix.shape[0] != adj_matrix.shape[1]:
                st.sidebar.error("The adjacency matrix must be square.")
            elif adj_matrix.shape[0] != n_oscillators:
                st.sidebar.error(f"Matrix dimensions ({adj_matrix.shape[0]}x{adj_matrix.shape[1]}) don't match oscillator count ({n_oscillators}).")
            else:
                st.sidebar.success("Adjacency matrix validated successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing matrix: {str(e)}")
            adj_matrix = None

# Create tabs for different visualizations (Network is default tab)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Network", "Distributions", "Animation", "Database", "Configurations"])

# Function to simulate model
@st.cache_data(ttl=300)
def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step, random_seed, adjacency_matrix=None):
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed,
        adjacency_matrix=adjacency_matrix
    )
    
    times, phases, order_parameter = model.simulate()
    return model, times, phases, order_parameter

# Run the simulation
model, times, phases, order_parameter = run_simulation(
    n_oscillators=n_oscillators,
    coupling_strength=coupling_strength,
    frequencies=frequencies,
    simulation_time=simulation_time,
    time_step=time_step,
    random_seed=random_seed,
    adjacency_matrix=adj_matrix
)

########################
# TAB 1: NETWORK TAB
########################
with tab1:
    st.markdown("<h2 class='gradient_text2'>Network Structure</h2>", unsafe_allow_html=True)
    
    # Display simulation information
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a network visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    # Import networkx for graph visualization
    import networkx as nx
    
    # Make sure adj_matrix is defined for all network types
    if network_type == "All-to-All":
        # For all-to-all, create a fully connected matrix with uniform coupling
        network_adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(network_adj_matrix, 0)  # No self-connections
    elif network_type == "Nearest Neighbor":
        # For nearest neighbor, create a ring topology
        network_adj_matrix = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            # Connect to left and right neighbors on the ring
            network_adj_matrix[i, (i-1) % n_oscillators] = 1
            network_adj_matrix[i, (i+1) % n_oscillators] = 1
    elif network_type == "Random":
        # For random, create random connections with 20% probability
        np.random.seed(random_seed)  # Use same seed for reproducibility
        network_adj_matrix = np.random.random((n_oscillators, n_oscillators)) < 0.2
        network_adj_matrix = network_adj_matrix.astype(float)
        np.fill_diagonal(network_adj_matrix, 0)  # No self-connections
    else:  # Custom Adjacency Matrix
        network_adj_matrix = adj_matrix if adj_matrix is not None else np.ones((n_oscillators, n_oscillators))
    
    # Create a graph visualization using networkx
    G = nx.from_numpy_array(network_adj_matrix)
    
    # Create custom colormap that matches our gradient_text1 theme for nodes
    custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                ["#14a5ff", "#8138ff"], 
                                                N=256)
    
    # Sort oscillators by their natural frequency for consistent coloring
    # This is the SAME color assignment used throughout all visualizations
    sorted_indices = np.argsort(frequencies)
    color_indices = np.linspace(0, 1, n_oscillators)
    oscillator_colors = np.zeros(n_oscillators, dtype=object)
    
    # Assign colors based on frequency order
    for i, idx in enumerate(sorted_indices):
        oscillator_colors[idx] = custom_cmap(color_indices[i])
    
    # Choose layout based on network type
    if network_type == "Nearest Neighbor":
        # Circular layout for nearest neighbor (ring)
        pos = nx.circular_layout(G)
    elif n_oscillators <= 20:
        # Spring layout for smaller networks
        pos = nx.spring_layout(G, seed=random_seed)
    else:
        # Circular layout is better for visualization with many nodes
        pos = nx.circular_layout(G)
    
    # Create graph visualization
    ax1.set_facecolor('#121212')
    
    # Draw the graph
    edges = nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.5, 
                               edge_color='#00ffee', width=1.5)
    
    # Convert the RGBA colors to hex for networkx
    node_colors = []
    for c in oscillator_colors:
        # Create hex color from the custom colormap colors
        if hasattr(c, 'tolist'):  # If it's a numpy array
            rgba = c.tolist()
        else:  # If it's already a tuple/list
            rgba = c
        # Format as hex
        node_colors.append(f"#{int(rgba[0]*255):02x}{int(rgba[1]*255):02x}{int(rgba[2]*255):02x}")
    
    # Calculate dynamic node size based on number of nodes
    # Larger size for fewer nodes, smaller size for many nodes
    n_nodes = len(G.nodes)
    node_size = max(100, int(1000 * (1 / (0.1 * n_nodes + 0.5))))  # Formula for scaling
    
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, 
                               node_color=node_colors, 
                               node_size=node_size, alpha=0.9, 
                               edgecolors='white', linewidths=1.5)
    
    # Add node labels only if there are relatively few nodes
    if n_oscillators <= 15:
        labels = {i: str(i) for i in range(n_oscillators)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, 
                           font_color='white', font_weight='bold')
        
    # Add title and styling
    ax1.set_title(f'Oscillator Network Graph ({network_type})', 
               color='white', fontsize=14, pad=15)
    ax1.set_axis_off()
    
    # Add a legend explaining node colors
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.1), 
                markeredgecolor='white', markersize=10, label='Lower frequency'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.5), 
                markeredgecolor='white', markersize=10, label='Medium frequency'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.9), 
                markeredgecolor='white', markersize=10, label='Higher frequency')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', 
            frameon=True, framealpha=0.7, facecolor='#121212', 
            edgecolor='#555555', labelcolor='white')
    
    # Create a heatmap of the adjacency matrix on the right side
    im = ax2.imshow(network_adj_matrix, cmap='viridis')
    plt.colorbar(im, ax=ax2, label='Connection Strength')
    
    # Add labels and styling
    ax2.set_title(f'Adjacency Matrix', color='white', fontsize=14)
    ax2.set_xlabel('Oscillator Index', color='white')
    ax2.set_ylabel('Oscillator Index', color='white')
    
    # Set background color
    ax2.set_facecolor('#1a1a1a')
    fig.patch.set_facecolor('#121212')
    
    # Add a grid to help distinguish cells
    ax2.grid(False)
    
    # Add text annotations for connection strength (only for matrices smaller than 15x15)
    if n_oscillators <= 12:
        for i in range(network_adj_matrix.shape[0]):
            for j in range(network_adj_matrix.shape[1]):
                if network_adj_matrix[i, j] > 0:
                    ax2.text(j, i, f"{network_adj_matrix[i, j]:.1f}", 
                          ha="center", va="center", 
                          color="white" if network_adj_matrix[i, j] < 0.7 else "black",
                          fontsize=9)
    
    # Adjust spacing between subplots
    plt.tight_layout()
    
    # Display the figure
    st.pyplot(fig)
    
    st.markdown("""
    <div class='section'>
        <p>The network visualization shows:</p>
        <ul>
            <li><b>Left:</b> Graph representation of oscillator connections, with nodes colored by natural frequency</li>
            <li><b>Right:</b> Adjacency matrix representation, where each cell (i,j) represents the connection strength between oscillators</li>
        </ul>
        <p>The structure of this network affects how synchronization patterns emerge and propagate through the system.</p>
    </div>
    """, unsafe_allow_html=True)

########################
# TAB 2: DISTRIBUTIONS TAB
########################
with tab2:
    st.markdown("<h2 class='gradient_text2'>Initial Distributions</h2>", unsafe_allow_html=True)
    
    # Display simulation information at the top
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Add the distribution histograms
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.markdown("<h4 class='gradient_text2'>Natural Frequency Distribution</h4>", unsafe_allow_html=True)
        
        # Create frequency distribution histogram
        fig_freq, ax_freq = plt.subplots(figsize=(3.5, 2.5))
        
        # Use a gradient colormap for the histogram
        n_bins = 15
        counts, bin_edges = np.histogram(frequencies, bins=n_bins)
        
        # Create custom colormap that matches our gradient theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                ["#00ffee", "#27aaff"], 
                                                N=256)
        
        # Create custom colors with a gradient effect that matches our theme
        colors = custom_cmap(np.linspace(0.1, 0.9, n_bins))
        
        # Plot the histogram with gradient colors and outline
        bars = ax_freq.bar(
            (bin_edges[:-1] + bin_edges[1:]) / 2, 
            counts, 
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color=colors, 
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add a soft glow effect behind bars
        for bar, color in zip(bars, colors):
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            glow = plt.Rectangle(
                (x - width * 0.05, 0), 
                width * 1.1, 
                height, 
                color=color,
                alpha=0.3,
                zorder=-1
            )
            ax_freq.add_patch(glow)
        
        # Enhance the axes and labels
        ax_freq.set_facecolor('#1a1a1a')
        ax_freq.set_xlabel('Natural Frequency', fontsize=12, fontweight='bold', color='white')
        ax_freq.set_ylabel('Count', fontsize=12, fontweight='bold', color='white')
        ax_freq.set_title(f'Natural Frequency Distribution ({freq_type})', 
                       fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Add mean frequency marker
        mean_freq = np.mean(frequencies)
        ax_freq.axvline(x=mean_freq, color='#ff5555', linestyle='-', linewidth=2, alpha=0.7,
                      label=f'Mean: {mean_freq:.2f}')
        ax_freq.legend(framealpha=0.7)
        
        # Customize grid
        ax_freq.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Add a subtle box around the plot
        for spine in ax_freq.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        st.pyplot(fig_freq)
        
        # Distribution properties description
        st.markdown(f"""
        <div class='section' style='font-size: 0.85em;'>
            <p><b>Mean:</b> {np.mean(frequencies):.4f}</p>
            <p><b>Standard Deviation:</b> {np.std(frequencies):.4f}</p>
            <p><b>Min:</b> {np.min(frequencies):.4f}</p>
            <p><b>Max:</b> {np.max(frequencies):.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with dist_col2:
        st.markdown("<h4 class='gradient_text2'>Initial Phase Distribution</h4>", unsafe_allow_html=True)
        
        # Create initial phase distribution histogram
        fig_init_phase, ax_init_phase = plt.subplots(figsize=(3.5, 2.5))
        
        initial_phases = phases[:, 0] % (2 * np.pi)
        
        # Use a gradient colormap for the histogram
        n_bins = 15
        counts, bin_edges = np.histogram(initial_phases, bins=n_bins)
        
        # Create custom colors with a gradient effect that matches our theme
        colors = custom_cmap(np.linspace(0.1, 0.9, n_bins))
        
        # Plot the histogram with gradient colors and outline
        bars = ax_init_phase.bar(
            (bin_edges[:-1] + bin_edges[1:]) / 2, 
            counts, 
            width=(bin_edges[1] - bin_edges[0]) * 0.9,
            color=colors, 
            alpha=0.8,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add a soft glow effect behind bars
        for bar, color in zip(bars, colors):
            x = bar.get_x()
            width = bar.get_width()
            height = bar.get_height()
            glow = plt.Rectangle(
                (x - width * 0.05, 0), 
                width * 1.1, 
                height, 
                color=color,
                alpha=0.3,
                zorder=-1
            )
            ax_init_phase.add_patch(glow)
        
        # Enhance the axes and labels
        ax_init_phase.set_facecolor('#1a1a1a')
        ax_init_phase.set_xlabel('Phase (mod 2π)', fontsize=12, fontweight='bold', color='white')
        ax_init_phase.set_ylabel('Count', fontsize=12, fontweight='bold', color='white')
        ax_init_phase.set_title('Initial Phase Distribution (t=0)', 
                             fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Calculate initial order parameter
        initial_r = order_parameter[0]
        initial_psi = np.angle(np.sum(np.exp(1j * initial_phases))) % (2 * np.pi)
        
        # Add mean phase marker if order parameter is significant
        if initial_r > 0.3:
            ax_init_phase.axvline(x=initial_psi, color='#ff5555', linestyle='-', linewidth=2, alpha=0.7,
                               label=f'Mean Phase: {initial_psi:.2f}')
            ax_init_phase.legend(framealpha=0.7)
        
        # Customize grid
        ax_init_phase.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Add a subtle box around the plot
        for spine in ax_init_phase.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        st.pyplot(fig_init_phase)
        
        # Initial phase properties description
        st.markdown(f"""
        <div class='section' style='font-size: 0.85em;'>
            <p><b>Initial Order Parameter:</b> {initial_r:.4f}</p>
            <p><b>Initial Mean Phase:</b> {initial_psi:.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Order Parameter Analysis section
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Order Parameter Analysis</h3>
        <p>The order parameter r(t) measures the degree of synchronization among oscillators:</p>
        <ul>
            <li>r = 1: Complete synchronization (all oscillators have the same phase)</li>
            <li>r = 0: Complete desynchronization (phases are uniformly distributed)</li>
        </ul>
        <p>At critical coupling strength (K_c), the system transitions from desynchronized to partially synchronized state.</p>
    </div>
    """, unsafe_allow_html=True)

########################
# TAB 3: ANIMATION TAB
########################
with tab3:
    st.markdown("<h2 class='gradient_text2'>Interactive Animation</h2>", unsafe_allow_html=True)
    
    # Display simulation information at the top
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create oscillator visualization
    # Removed "Oscillator Synchronization" header as requested
    
    # Initialize time_index in session state if not present
    if 'time_index' not in st.session_state:
        st.session_state.time_index = 0
        
    # Set time index from session state
    time_index = st.session_state.time_index
    
    # Initialize animation variables
    animate = False
    animation_speed = 3.0
    
    # Get time and order parameter values for the current time index
    current_time = times[time_index]
    current_r = order_parameter[time_index]
    
    # Import needed module
    from matplotlib.collections import LineCollection
    
    # Function to create the phase visualization
    def create_phase_plot(time_idx):
        # Create visualization with enhanced visuals for dark theme
        fig_circle = plt.figure(figsize=(5, 5))
        ax_circle = fig_circle.add_subplot(111)
        
        # Add background glow effect
        bg_circle = plt.Circle((0, 0), 1.1, fill=True, color='#121a24', alpha=0.6, zorder=1)
        ax_circle.add_patch(bg_circle)
        
        # Add subtle circle rings for reference in a complementary orange color
        for radius in [0.25, 0.5, 0.75]:
            ring = plt.Circle((0, 0), radius, fill=False, color='#aa6633', 
                            linestyle=':', alpha=0.4, zorder=2)
            ax_circle.add_patch(ring)
        
        # Draw unit circle with glow effect - using light orange that complements the arrow
        circle_glow = plt.Circle((0, 0), 1.02, fill=False, color='#ffaa66', alpha=0.3, linewidth=3, zorder=3)
        ax_circle.add_patch(circle_glow)
        
        # Main unit circle in light orange
        circle = plt.Circle((0, 0), 1, fill=False, color='#ff9955', linestyle='-', 
                         linewidth=1.5, alpha=0.8, zorder=4)
        ax_circle.add_patch(circle)
        
        # Plot oscillators
        phases_at_time = phases[:, time_idx]
        x = np.cos(phases_at_time)
        y = np.sin(phases_at_time)
        
        # Create custom colormap that matches our gradient_text1 theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                     ["#14a5ff", "#8138ff"], 
                                                     N=256)
        
        # Sort oscillators by their natural frequency
        sorted_indices = np.argsort(frequencies)
        # Create colors based on frequency ordering
        color_indices = np.linspace(0, 1, n_oscillators)
        colors = [custom_cmap(idx) for idx in color_indices]
        
        # Map colors to oscillators by frequency order
        oscillator_colors = np.zeros((n_oscillators, 4))  # RGBA colors
        for i, idx in enumerate(sorted_indices):
            oscillator_colors[idx] = colors[i]
        
        # Scatter plot with oscillator colors and brighter versions for outlines
        # Create brighter versions of the colors for outlines
        bright_oscillator_colors = np.copy(oscillator_colors)
        for i in range(len(bright_oscillator_colors)):
            # Make RGB values brighter (closer to white) but preserve alpha
            bright_oscillator_colors[i, :3] = np.minimum(1.0, bright_oscillator_colors[i, :3] * 1.5)  # 50% brighter
        
        # Use custom edge colors instead of white
        sc = ax_circle.scatter(x, y, facecolors=oscillator_colors, edgecolors=bright_oscillator_colors, s=180, 
                             alpha=0.9, linewidth=1.0, zorder=10)
        
        # Calculate and show order parameter
        r = order_parameter[time_idx]
        psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
        
        # Draw arrow showing mean field with glow effect - using orange color
        # First add glow/shadow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.07, head_length=0.12, fc='#ffaa33', ec='#ffaa33', 
                       width=0.03, alpha=0.3, zorder=5)
        
        # Then add main arrow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.05, head_length=0.1, fc='#ff9500', ec='#ff9500', 
                       width=0.02, zorder=6)
        
        # Draw axes
        ax_circle.axhline(y=0, color='#555555', linestyle='-', alpha=0.5, zorder=0)
        ax_circle.axvline(x=0, color='#555555', linestyle='-', alpha=0.5, zorder=0)
        
        ax_circle.set_xlim(-1.2, 1.2)
        ax_circle.set_ylim(-1.2, 1.2)
        ax_circle.set_aspect('equal')
        
        # Add subtle grid
        ax_circle.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Enhance title
        ax_circle.set_title(f'Oscillators at t={times[time_idx]:.2f}', 
                          color='white', fontsize=14, pad=15)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
        
        return fig_circle
    
    # Function to create oscillator phases over time plot (as dots)
    def create_oscillator_phases_plot(time_idx):
        fig, ax = plt.subplots(figsize=(12, 3.5))
        
        # Add background
        ax.set_facecolor('#1a1a1a')
        
        # Create transparent bands at phase regions
        for y in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
            ax.axhspan(y-0.1, y+0.1, color='#222233', alpha=0.4, zorder=0)
        
        # Create custom colormap that matches our gradient_text1 theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                     ["#14a5ff", "#8138ff"], 
                                                     N=256)
        
        # Sort oscillators by their natural frequency for consistent coloring
        sorted_indices = np.argsort(frequencies)
        color_indices = np.linspace(0, 1, n_oscillators)
        oscillator_colors = np.zeros(n_oscillators, dtype=object)
        
        # Assign colors based on frequency order
        for i, idx in enumerate(sorted_indices):
            oscillator_colors[idx] = custom_cmap(color_indices[i])
        
        # Plot all oscillators as dots up to the current time point
        for i in range(n_oscillators):
            color = oscillator_colors[i]
            
            # Create a brighter version of the color for edge
            # Extract RGB values and make them brighter
            rgb = matplotlib.colors.to_rgb(color)
            # Create brighter version (closer to white)
            bright_color = tuple(min(1.0, c * 1.5) for c in rgb)
            
            # Plot oscillator phases as filled dots with color gradient
            ax.scatter(times[:time_idx+1], phases[i, :time_idx+1] % (2 * np.pi), 
                     facecolors=color, edgecolor=bright_color, alpha=0.7, s=50, 
                     linewidth=0.5, zorder=5)
            
            # Add a subtle connecting line with low opacity
            ax.plot(times[:time_idx+1], phases[i, :time_idx+1] % (2 * np.pi), 
                  color=color, alpha=0.2, linewidth=0.8, zorder=2)
            
            # Highlight current position with a larger filled marker
            ax.scatter([times[time_idx]], [phases[i, time_idx] % (2 * np.pi)], 
                     s=140, facecolors=color, edgecolor=bright_color, 
                     linewidth=1.0, zorder=15)
        
        # Add labels for key phase positions
        phase_labels = [(0, '0'), (np.pi/2, 'π/2'), (np.pi, 'π'), (3*np.pi/2, '3π/2'), (2*np.pi, '2π')]
        for y, label in phase_labels:
            ax.annotate(label, xy=(-0.02, y), xycoords=('axes fraction', 'data'),
                      fontsize=11, color='white', ha='center', va='center')
        
        # Plot styling
        ax.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel('Phase (mod 2π)', fontsize=13, fontweight='bold', color='white')
        ax.set_title(f'Oscillator Phases at t={times[time_idx]:.2f}', 
                   fontsize=14, fontweight='bold', color='white', pad=15)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_xlim(times.min(), times.max())
        
        # Custom grid
        ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Add box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
            
        return fig
    
    # Create a function to create order parameter plot over time (as a dot plot)
    def create_order_parameter_plot(time_idx):
        fig, ax = plt.subplots(figsize=(5, 5))
        
        # Add background gradient
        ax.set_facecolor('#1a1a1a')
        
        # Add subtle horizontal bands for visual reference
        for y in np.linspace(0, 1, 6):
            ax.axhspan(y-0.05, y+0.05, color='#222233', alpha=0.3, zorder=0)
        
        # Create a custom colormap that matches the blue gradient used in the sidebar
        cmap = LinearSegmentedColormap.from_list("order_param", 
                                             ["#00e8ff", "#14b5ff", "#3a98ff", "#0070eb"], 
                                             N=256)
        
        # Plot order parameter with filled gradient dots and brighter outline
        base_colors = [cmap(r) for r in order_parameter[:time_idx+1]]
        edge_colors = []
        
        # Create brighter versions of each color for the outlines
        for color in base_colors:
            rgb = matplotlib.colors.to_rgb(color)
            # Make RGB values brighter but preserve alpha
            bright_color = tuple(min(1.0, c * 1.5) for c in rgb)
            edge_colors.append(bright_color)
        
        scatter = ax.scatter(times[:time_idx+1], order_parameter[:time_idx+1],
                          facecolors=base_colors, edgecolors=edge_colors,
                          s=70, alpha=0.9, linewidth=0.5, zorder=10)
        
        # Add a connecting line with low opacity
        ax.plot(times[:time_idx+1], order_parameter[:time_idx+1], 
              color='white', alpha=0.3, linewidth=1, zorder=5)
        
        # Highlight current position with a larger filled marker
        if time_idx > 0:
            # Get color and make brighter version for outline
            current_color = cmap(order_parameter[time_idx])
            rgb = matplotlib.colors.to_rgb(current_color)
            bright_current = tuple(min(1.0, c * 1.5) for c in rgb)
            
            ax.scatter([times[time_idx]], [order_parameter[time_idx]], 
                     s=180, facecolors=current_color, 
                     edgecolors=bright_current, 
                     linewidth=1.0, zorder=15)
        
        # Add highlights at important thresholds
        ax.axhline(y=0.5, color='#aaaaaa', linestyle='--', alpha=0.5, zorder=1, 
                 label='Partial Synchronization (r=0.5)')
        ax.axhline(y=0.8, color='#ffffff', linestyle='--', alpha=0.5, zorder=1,
                 label='Strong Synchronization (r=0.8)')
        
        # Enhance the plot appearance
        ax.set_xlim(times.min(), times.max())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel('Order Parameter r(t)', fontsize=13, fontweight='bold', color='white')
        ax.set_title(f'Phase Synchronization at t={times[time_idx]:.2f}', 
                   fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Create custom grid
        ax.grid(True, color='#333333', alpha=0.5, linestyle=':')
        
        # Legend removed as requested
        
        # Add subtle box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
            
        return fig
    
    # Create a more space-efficient layout 
    # First row: Oscillator phases plot (wider view)
    phases_plot_placeholder = st.empty()
    phases_plot_placeholder.pyplot(create_oscillator_phases_plot(time_index))
    
    # Second row: Circle plot and order parameter side by side
    col1, col2 = st.columns(2)
    with col1:
        circle_plot_placeholder = st.empty()
        circle_plot_placeholder.pyplot(create_phase_plot(time_index))
    
    with col2:
        order_plot_placeholder = st.empty()
        order_plot_placeholder.pyplot(create_order_parameter_plot(time_index))
    
    # Removed "Current Time" display as requested
    
    # Playback controls layout - now after the plots
    playback_container = st.container()
    
    # Add a slider to manually control visualization time point
    time_index = playback_container.slider(
        "Time Point", 
        min_value=0, 
        max_value=len(times)-1, 
        value=st.session_state.time_index,
        help="Manually select a specific time point to display"
    )
    st.session_state.time_index = time_index
    
    # Centered animation controls with custom styling - removed speed slider
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    # Set a fixed animation speed value
    animation_speed = 3.0  # Fixed moderate animation speed
    
    # Previous frame button
    if col1.button("⏮ Previous", use_container_width=True):
        if st.session_state.time_index > 0:
            st.session_state.time_index -= 1
            st.rerun()
    
    # Play Animation button
    animate = col2.button("▶ Play", use_container_width=True)
    
    # Next frame button
    if col3.button("⏭ Next", use_container_width=True):
        if st.session_state.time_index < len(times) - 1:
            st.session_state.time_index += 1
            st.rerun()
    
    # Save simulation button
    if col4.button("💾 Save", use_container_width=True, type="primary"):
        # Get frequency parameters based on the selected distribution
        freq_params = {}
        if freq_type == "Normal":
            freq_params = {"mean": freq_mean, "std": freq_std}
        elif freq_type == "Uniform":
            freq_params = {"min": freq_min, "max": freq_max}
        elif freq_type == "Bimodal":
            freq_params = {"peak1": peak1, "peak2": peak2}
        elif freq_type == "Custom":
            freq_params = {"values": custom_freqs}
        
        # Save to database
        sim_id = store_simulation(
            model=model,
            times=times,
            phases=phases,
            order_parameter=order_parameter,
            frequencies=frequencies,
            freq_type=freq_type,
            freq_params=freq_params,
            adjacency_matrix=adj_matrix
        )
        
        if sim_id:
            st.success(f"Simulation data saved successfully with ID: {sim_id}")
            st.info("You can access this data in the Database tab.")
        else:
            st.error("Failed to save simulation data.")
            
    # If animation is triggered
    if animate:
        # Get the current time index as the starting point
        start_idx = st.session_state.time_index
        
        # Calculate how many frames to skip based on speed
        frame_skip = max(1, int(11 - animation_speed))
        
        # Set up a progress bar
        progress_bar = st.progress(0)
        
        # Animation loop
        for i in range(start_idx, len(times), frame_skip):
            # Update the session state
            st.session_state.time_index = i
            
            # Update progress bar
            progress = (i - start_idx) / (len(times) - 1 - start_idx) if i < len(times) - 1 else 1.0
            progress_bar.progress(progress)
            
            # Update all three plots
            circle_plot_placeholder.pyplot(create_phase_plot(i))
            phases_plot_placeholder.pyplot(create_oscillator_phases_plot(i))
            order_plot_placeholder.pyplot(create_order_parameter_plot(i))
            
            # Add a short pause to control animation speed
            time.sleep(0.1 / animation_speed)
        
        # Clear progress bar after animation
        progress_bar.empty()
    
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Visualization Guide</h3>
        <p>The <b>top plot</b> shows oscillator phases over time. Each horizontal trace represents one oscillator's phase trajectory with consistent coloring based on the oscillator's natural frequency.</p>
        <p>The <b>bottom left plot</b> shows oscillators on a unit circle. Each colored dot represents an oscillator at its current phase position. The orange arrow shows the mean field vector, with length equal to the order parameter r.</p>
        <p>The <b>bottom right plot</b> shows the order parameter over time, with color-coded dots showing the synchronization level from 0 (no synchronization) to 1 (complete synchronization).</p>
        <p>Click "Play Animation" to watch all three visualizations animate together to see the synchronization process in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Order Parameter Analysis</h3>
        <p>The order parameter r(t) measures the degree of synchronization among oscillators:</p>
        <ul>
            <li>r = 1: Complete synchronization (all oscillators have the same phase)</li>
            <li>r = 0: Complete desynchronization (phases are uniformly distributed)</li>
        </ul>
        <p>At critical coupling strength (K_c), the system transitions from desynchronized to partially synchronized state.</p>
    </div>
    """, unsafe_allow_html=True)

########################
# TAB 4: DATABASE TAB
########################
with tab4:
    st.markdown("<h2 class='gradient_text2'>Database</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Store and Retrieve Simulations</h3>
        <p>Save your simulation parameters and results to the database for future reference and comparison.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for saving and loading
    db_col1, db_col2 = st.columns(2)
    
    with db_col1:
        st.markdown("<h4 class='gradient_text1'>Save Current Simulation</h4>", unsafe_allow_html=True)
        
        # Get parameters of frequency distribution based on selected type
        freq_params = {}
        if freq_type == "Normal":
            freq_params = {"mean": freq_mean, "std": freq_std}
        elif freq_type == "Uniform":
            freq_params = {"min": freq_min, "max": freq_max}
        elif freq_type == "Bimodal":
            freq_params = {"peak1": peak1, "peak2": peak2}
        elif freq_type == "Custom":
            freq_params = {"custom_values": custom_freqs}
        
        # Button to save current simulation
        if st.button("Save to Database", use_container_width=True):
            # Store the simulation in the database
            sim_id = store_simulation(
                model=model, 
                times=times, 
                phases=phases, 
                order_parameter=order_parameter,
                frequencies=frequencies,
                freq_type=freq_type,
                freq_params=freq_params,
                adjacency_matrix=adj_matrix  # Store custom adjacency matrix if provided
            )
            
            if sim_id:
                st.success(f"Simulation saved successfully with ID: {sim_id}")
            else:
                st.error("Failed to save simulation to database.")
    
    with db_col2:
        st.markdown("<h4 class='gradient_text1'>Load Saved Simulations</h4>", unsafe_allow_html=True)
        
        # Get list of saved simulations
        simulations = list_simulations()
        
        if not simulations:
            st.info("No saved simulations found in the database.")
        else:
            # Create a selection box for the simulations
            sim_options = [f"ID: {sim['id']} - {sim['n_oscillators']} oscillators, K={sim['coupling_strength']}, {sim['timestamp'].strftime('%Y-%m-%d %H:%M')}" 
                         for sim in simulations]
            
            selected_sim = st.selectbox("Select a simulation to load", sim_options)
            
            if selected_sim:
                # Extract simulation ID from selection
                sim_id = int(selected_sim.split("ID: ")[1].split(" -")[0])
                
                if st.button("Load Simulation", use_container_width=True):
                    # Load the simulation from the database
                    sim_data = get_simulation(sim_id)
                    
                    if sim_data:
                        # Display basic information about the loaded simulation
                        st.markdown(f"""
                        <div class='section'>
                            <h4 class='gradient_text2'>Simulation Details</h4>
                            <p><b>ID:</b> {sim_data['id']}</p>
                            <p><b>Timestamp:</b> {sim_data['timestamp']}</p>
                            <p><b>Oscillators:</b> {sim_data['n_oscillators']}</p>
                            <p><b>Coupling Strength:</b> {sim_data['coupling_strength']}</p>
                            <p><b>Frequency Distribution:</b> {sim_data['frequency_distribution']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create visualization of the loaded simulation
                        st.markdown("<h4 class='gradient_text2'>Order Parameter</h4>", unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(7, 3.5))
                        ax.set_facecolor('#1a1a1a')
                        
                        # Create custom colormap
                        cmap = LinearSegmentedColormap.from_list("order_param", 
                                                            ["#00ffee", "#27aaff"], 
                                                            N=256)
                        
                        # Plot order parameter
                        ax.plot(range(len(sim_data['order_parameter']['r'])), 
                              sim_data['order_parameter']['r'], 
                              color=cmap(0.5), linewidth=3)
                        
                        ax.set_xlabel('Time Index', fontsize=12, fontweight='bold', color='white')
                        ax.set_ylabel('Order Parameter r(t)', fontsize=12, fontweight='bold', color='white')
                        ax.set_title('Phase Synchronization', fontsize=14, fontweight='bold', color='white')
                        ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
                        
                        # Add a subtle box around the plot
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#555555')
                            spine.set_linewidth(1)
                        
                        st.pyplot(fig)
                    else:
                        st.error("Failed to load simulation from database.")
    
    # Add section for database management
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Database Management</h3>
        <p>Manage your saved simulations and keep your database organized.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all simulations in a table
    if simulations:
        st.markdown("<h4 class='gradient_text2'>Saved Simulations</h4>", unsafe_allow_html=True)
        
        # Create a DataFrame for display
        import pandas as pd
        sim_df = pd.DataFrame([
            {
                "ID": sim["id"],
                "Date": sim["timestamp"].strftime("%Y-%m-%d"),
                "Time": sim["timestamp"].strftime("%H:%M:%S"),
                "Oscillators": sim["n_oscillators"],
                "Coupling": f"{sim['coupling_strength']:.2f}",
                "Distribution": sim["frequency_distribution"]
            } for sim in simulations
        ])
        
        st.dataframe(sim_df, use_container_width=True)
        
        # Delete simulation functionality
        st.markdown("<h4 class='gradient_text2'>Delete Simulation</h4>", unsafe_allow_html=True)
        
        delete_id = st.number_input("Enter Simulation ID to Delete", min_value=1, step=1)
        
        if st.button("Delete", use_container_width=True):
            # Confirm before deletion
            if delete_id in [sim["id"] for sim in simulations]:
                success = delete_simulation(delete_id)
                if success:
                    st.success(f"Simulation with ID {delete_id} was deleted successfully.")
                    st.rerun()  # Refresh the page to update the list
                else:
                    st.error("Failed to delete simulation.")
            else:
                st.warning(f"No simulation found with ID {delete_id}.")

# Footer removed as requested

########################
# TAB 5: CONFIGURATIONS TAB
########################
with tab5:
    st.markdown("<h2 class='gradient_text2'>Save & Load Configurations</h2>", unsafe_allow_html=True)
    
    # Display description
    st.markdown("""
    <div style='background-color: rgba(0,0,0,0.3); padding: 15px; border-radius: 10px; margin-bottom: 20px;'>
        <p>Save your current simulation configuration for future experiments. This preserves all parameters including:</p>
        <ul>
            <li>Number of oscillators and coupling strength</li>
            <li>Frequency distribution settings</li>
            <li>Network connectivity type and structure</li>
            <li>Simulation time and step size</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Create columns for save and load panels
    save_col, load_col = st.columns(2)
    
    # SAVE CONFIGURATION PANEL
    with save_col:
        st.markdown("<h3 class='gradient_text1'>Save Current Configuration</h3>", unsafe_allow_html=True)
        
        # Configuration name input
        config_name = st.text_input("Configuration Name", placeholder="Enter a unique name")
        
        # Get frequency parameters based on the selected distribution
        freq_params = {}
        if freq_type == "Normal":
            freq_params = {"mean": freq_mean, "std": freq_std}
        elif freq_type == "Uniform":
            freq_params = {"min": freq_min, "max": freq_max}
        elif freq_type == "Bimodal":
            freq_params = {"peak1": peak1, "peak2": peak2}
        elif freq_type == "Custom":
            freq_params = {"values": custom_freqs}
        
        # Save button
        if st.button("Save Configuration", type="primary"):
            if not config_name:
                st.error("Please enter a configuration name.")
            else:
                # Determine which adjacency matrix to save
                if network_type == "Custom Adjacency Matrix":
                    save_adj_matrix = adj_matrix
                elif network_type == "All-to-All":
                    save_adj_matrix = np.ones((n_oscillators, n_oscillators))
                    np.fill_diagonal(save_adj_matrix, 0)  # No self-connections
                elif network_type == "Nearest Neighbor":
                    save_adj_matrix = np.zeros((n_oscillators, n_oscillators))
                    for i in range(n_oscillators):
                        save_adj_matrix[i, (i-1) % n_oscillators] = 1
                        save_adj_matrix[i, (i+1) % n_oscillators] = 1
                elif network_type == "Random":
                    np.random.seed(random_seed)
                    save_adj_matrix = np.random.random((n_oscillators, n_oscillators)) < 0.2
                    save_adj_matrix = save_adj_matrix.astype(float)
                    np.fill_diagonal(save_adj_matrix, 0)
                
                # Save to database
                try:
                    config_id = save_configuration(
                        name=config_name,
                        n_oscillators=n_oscillators,
                        coupling_strength=coupling_strength,
                        simulation_time=simulation_time,
                        time_step=time_step,
                        random_seed=random_seed,
                        network_type=network_type,
                        frequency_distribution=freq_type,
                        frequency_params=freq_params,
                        adjacency_matrix=save_adj_matrix
                    )
                    
                    if config_id:
                        st.success(f"Configuration '{config_name}' saved successfully!")
                    else:
                        st.error(f"A configuration with the name '{config_name}' already exists.")
                except Exception as e:
                    st.error(f"Error saving configuration: {str(e)}")
    
    # LOAD CONFIGURATION PANEL
    with load_col:
        st.markdown("<h3 class='gradient_text1'>Load Saved Configuration</h3>", unsafe_allow_html=True)
        
        # Get list of saved configurations
        configs = list_configurations()
        
        if not configs:
            st.info("No saved configurations found. Save a configuration first.")
        else:
            # Create a dropdown with configuration names
            config_options = {f"{c['name']} (Oscillators: {c['n_oscillators']}, K: {c['coupling_strength']})": c['id'] for c in configs}
            selected_config = st.selectbox("Select Configuration", options=list(config_options.keys()))
            
            # Load configuration when button is clicked
            if st.button("Load Configuration", type="primary"):
                selected_config_id = config_options[selected_config]
                config = get_configuration(selected_config_id)
                
                if config:
                    # Store the configuration in a special session state variable
                    # This will be used on the next rerun to set the widget values
                    st.session_state.loaded_config = config
                    
                    # Show success message and rerun the app to apply the configuration
                    st.success(f"Configuration '{config['name']}' loaded successfully! Applying settings...")
                    st.rerun()
                else:
                    st.error("Failed to load the selected configuration.")
            
            # Option to delete a configuration
            with st.expander("Delete Configurations"):
                st.warning("Warning: Deletion is permanent and cannot be undone.")
                delete_options = {f"{c['name']} (Created: {c['timestamp'].strftime('%Y-%m-%d %H:%M')})": c['id'] for c in configs}
                config_to_delete = st.selectbox("Select Configuration to Delete", options=list(delete_options.keys()))
                
                if st.button("Delete Selected Configuration", type="primary"):
                    delete_id = delete_options[config_to_delete]
                    if delete_configuration(delete_id):
                        st.success(f"Configuration deleted successfully.")
                        st.rerun()  # Refresh the page to update the list
                    else:
                        st.error("Failed to delete configuration.")