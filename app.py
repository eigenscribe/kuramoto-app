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
                  get_configuration_by_name, export_configuration_to_json, import_configuration_from_json)

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
            elif isinstance(matrix, list):
                # Convert list to numpy array if it's still a list
                matrix = np.array(matrix)
            
            # Make sure no self-loops (diagonal elements should be zero)
            # This is important to ensure consistent visualization
            if hasattr(matrix, 'shape') and matrix.shape[0] == matrix.shape[1]:
                np.fill_diagonal(matrix, 0)
                
            # Print debug info
            print(f"Loading adjacency matrix: type={type(matrix)}, shape={matrix.shape if hasattr(matrix, 'shape') else 'unknown'}")
            
            if hasattr(matrix, 'shape'):
                print(f"Matrix sum: {np.sum(matrix)}, non-zeros: {np.count_nonzero(matrix)}")
                if matrix.shape[0] >= 3:
                    print(f"Sample (top-left 3x3):")
                    print(matrix[:3, :3])
                
            # Convert matrix to string representation for the text area
            matrix_str = ""
            for row in matrix:
                matrix_str += ", ".join(str(val) for val in row) + "\n"
            st.session_state.adj_matrix_input = matrix_str.strip()
            
            # Store the matrix for later use in this session
            # This ensures the matrix is properly passed to the simulation
            # Always update the matrix in session state - force overwrite to ensure latest is used
            st.session_state.loaded_adj_matrix = matrix
            print(f"Stored adjacency matrix in session state with shape {matrix.shape}")
                
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
    # Create a default example matrix for a 5x5 ring topology
    default_matrix = "0, 1, 0, 0, 1\n1, 0, 1, 0, 0\n0, 1, 0, 1, 0\n0, 0, 1, 0, 1\n1, 0, 0, 1, 0"
    st.session_state.adj_matrix_input = default_matrix

# For auto-adjusting oscillator count based on matrix dimensions
if 'next_n_oscillators' not in st.session_state:
    st.session_state.next_n_oscillators = None
    
# If we have a pending oscillator count update from the previous run, apply it now
if st.session_state.next_n_oscillators is not None:
    print(f"Updating oscillator count from {st.session_state.n_oscillators} to {st.session_state.next_n_oscillators}")
    st.session_state.n_oscillators = st.session_state.next_n_oscillators
    st.session_state.next_n_oscillators = None  # Clear the pending update

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
# Check if we have a loaded adjacency matrix from a configuration
if 'loaded_adj_matrix' in st.session_state:
    adj_matrix = st.session_state.loaded_adj_matrix
    print(f"Retrieved adjacency matrix from session state with shape {adj_matrix.shape if hasattr(adj_matrix, 'shape') else 'unknown'}")
    
    # Safety check to ensure matrix is valid
    if hasattr(adj_matrix, 'shape') and adj_matrix.shape[0] > 0:
        print(f"Matrix looks valid: shape={adj_matrix.shape}, sum={np.sum(adj_matrix)}, non-zeros={np.count_nonzero(adj_matrix)}")
        
        # CRITICAL: We need to force the correct network type
        # This needs to take precedence over what's selected in the UI radio button
        if network_type != "Custom Adjacency Matrix":
            print("Detected loaded matrix with network type that doesn't match 'Custom Adjacency Matrix'.")
            print(f"Current network_type is '{network_type}' but will use matrix internally")
    else:
        print("Warning: Found loaded_adj_matrix in session state but it appears invalid:")
        print(f"Matrix type: {type(adj_matrix)}")
        if hasattr(adj_matrix, 'shape'):
            print(f"Shape: {adj_matrix.shape}")
        adj_matrix = None  # Reset to None if invalid matrix

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
    
    # Create a default example matrix if no prior matrix exists
    if not st.session_state.adj_matrix_input:
        # Create a simple default example for smaller number of oscillators
        default_matrix = ""
        for i in range(min(5, n_oscillators)):
            row = []
            for j in range(min(5, n_oscillators)):
                if i == j:
                    row.append("0")  # No self-connections
                elif abs(i-j) == 1 or abs(i-j) == min(5, n_oscillators)-1:  # Ring-neighbors
                    row.append("1")  # Connected
                else:
                    row.append("0")  # Not connected
            default_matrix += ", ".join(row) + "\n"
                
    # Make sure we have a non-empty value for the text area
    if network_type == "Custom Adjacency Matrix" and not st.session_state.adj_matrix_input:
        print("Custom matrix selected but no existing input - initializing default")
        st.session_state.adj_matrix_input = default_matrix
        
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
            
            # Ensure we have at least one row
            if len(rows) == 0:
                raise ValueError("No data found in matrix input")
                
            # Process each row, removing extra spaces and parsing values
            adj_matrix = []
            for row in rows:
                # Skip empty rows
                if not row.strip():
                    continue
                    
                # Process values in this row
                values = []
                for val in row.split(','):
                    # Convert to float, handling extra whitespace
                    cleaned_val = val.strip()
                    if cleaned_val:  # Skip empty entries
                        values.append(float(cleaned_val))
                
                # Ensure row has data
                if values:
                    adj_matrix.append(values)
            
            # Make sure we have a valid matrix with data
            if not adj_matrix:
                raise ValueError("Could not find valid numeric data in input")
                
            # Convert to numpy array for faster processing
            adj_matrix = np.array(adj_matrix)
            
            # Validate the adjacency matrix
            if adj_matrix.shape[0] != adj_matrix.shape[1]:
                st.sidebar.error(f"The adjacency matrix must be square. Current shape: {adj_matrix.shape}")
            elif adj_matrix.shape[0] != n_oscillators:
                # We can't modify widget session state once widgets are created,
                # so we'll save the desired dimension in a different session state variable
                matrix_dim = adj_matrix.shape[0]
                
                # Log information
                print(f"Matrix dimensions ({matrix_dim}) don't match current oscillator count ({n_oscillators})")
                
                # Store the matrix as is, don't try to resize it
                st.session_state.next_n_oscillators = matrix_dim
                
                # Show message explaining what's happening
                st.sidebar.info(f"""
                Matrix size ({matrix_dim}×{matrix_dim}) differs from current oscillator count ({n_oscillators}).
                The matrix will be used as-is for this simulation. Next time you interact with the UI, 
                the oscillator count will automatically update to match your matrix dimensions.
                """)
                
                # Keep local variable as is, use adj_matrix without modification
            else:
                st.sidebar.success("Adjacency matrix validated successfully!")
                
                # Add a dedicated button to force network visualization refresh
                if st.sidebar.button("🔄 Refresh", key="force_refresh_btn"):
                    st.session_state.refresh_network = True
                    print("Network refresh requested via button")
                    st.rerun()
                
                # Add save preset button and input field
                with st.sidebar.expander("Save as Preset"):
                    preset_name = st.text_input("Preset Name", key="preset_name", 
                                             placeholder="Enter a name for this matrix")
                    if st.button("💾 Save Preset", key="save_preset_btn"):
                        if preset_name:
                            # Save the configuration with current parameters
                            config_id = save_configuration(
                                name=preset_name,
                                n_oscillators=adj_matrix.shape[0],
                                coupling_strength=coupling_strength,
                                simulation_time=simulation_time,
                                time_step=time_step,
                                random_seed=random_seed,
                                network_type="Custom Adjacency Matrix",
                                frequency_distribution=freq_type,
                                frequency_params=json.dumps({
                                    "mean": float(freq_mean) if 'freq_mean' in locals() else 0.0,
                                    "std": float(freq_std) if 'freq_std' in locals() else 1.0,
                                    "min": float(freq_min) if 'freq_min' in locals() else -1.0,
                                    "max": float(freq_max) if 'freq_max' in locals() else 1.0
                                }),
                                adjacency_matrix=adj_matrix
                            )
                            st.success(f"Saved preset '{preset_name}' successfully!")
                            print(f"Saved matrix preset: '{preset_name}' with shape {adj_matrix.shape}")
                        else:
                            st.error("Please enter a preset name")
                
            # Store in session state for persistence
            st.session_state.loaded_adj_matrix = adj_matrix
            print(f"Updated adjacency matrix in session state with shape {adj_matrix.shape}")
                
        except Exception as e:
            st.sidebar.error(f"Error parsing matrix: {str(e)}")
            print(f"Matrix parsing error: {str(e)}")
            print(f"Input was: '{adj_matrix_input}'")
            adj_matrix = None

# Create tabs for different visualizations (Network is default tab)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Network", "Distributions", "Animation", "Database", "Configurations"])

# Set a unique key for each tab to force refresh of the Network tab
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Network"
    
# Add a hidden button that is programmatically clicked when checking a custom matrix
tab_key = f"tab_refresh_{int(time.time())}"
if st.session_state.current_tab == "Network" and adj_matrix is not None:
    # This ensures the network tab is refreshed when the adjacency matrix changes
    st.session_state.refresh_network = True

# Determine the effective network type for display and matrix creation
# If we have a custom matrix, ALWAYS force the network type to custom
# regardless of what's displayed in the UI
if adj_matrix is not None:
    print(f"Detected valid adjacency matrix - forcing internal network type to Custom")
    print(f"Matrix shape: {adj_matrix.shape}, sum: {np.sum(adj_matrix)}, non-zeros: {np.count_nonzero(adj_matrix)}")
    
    # Don't change the UI selection, but use Custom type for all internal processing
    network_type_internal = "Custom Adjacency Matrix"
    
    # CRITICAL: We must NOT delete the matrix from session state until we've used it successfully
    # Otherwise, it will be lost on the next rerun when Streamlit rebuilds the application
    
    # Keep a flag to signal we've saved the matrix for this session
    if 'using_loaded_matrix' not in st.session_state:
        st.session_state.using_loaded_matrix = True
        print("First use of loaded matrix - will keep in session state")
else:
    # Use the selected network type
    network_type_internal = network_type
    
    # If we don't have a custom matrix but the UI type is set to custom,
    # we need to ensure this is communicated clearly
    if network_type == "Custom Adjacency Matrix" and adj_matrix is None:
        # Use custom styled message with orange background instead of the default yellow warning
        # This matches the sidebar error messages styling
        st.markdown("""
        <div style="background-color: rgba(255,150,0,0.15); color: #ffaa50; 
                    padding: 10px; border-radius: 15px; border-left: 5px solid #ff8800;">
            <b>Matrix Input:</b> Please enter your custom adjacency matrix in the sidebar.
            The format should be comma-separated values with each row on a new line.
        </div>
        """, unsafe_allow_html=True)
        print("Warning: Custom adjacency matrix selected but no valid matrix found")

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
# If we have a custom adjacency matrix, adjust the oscillator count to match matrix dimensions
sim_n_oscillators = n_oscillators
if adj_matrix is not None:
    matrix_dim = adj_matrix.shape[0]
    if matrix_dim != n_oscillators:
        print(f"Adjusting simulation oscillator count to match matrix dimensions: {matrix_dim}")
        sim_n_oscillators = matrix_dim
        
        # Also adjust frequencies to match this count
        if len(frequencies) != sim_n_oscillators:
            print(f"Adjusting frequencies array from length {len(frequencies)} to {sim_n_oscillators}")
            if len(frequencies) > sim_n_oscillators:
                # Truncate if too many
                frequencies = frequencies[:sim_n_oscillators]
            else:
                # Extend by cycling through existing values if too few
                frequencies = np.resize(frequencies, sim_n_oscillators)

model, times, phases, order_parameter = run_simulation(
    n_oscillators=sim_n_oscillators,
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
    # Update current tab in session state to track which tab is active
    st.session_state.current_tab = "Network"
    
    # Force a reload when we have a custom matrix change
    if 'refresh_network' in st.session_state and st.session_state.refresh_network:
        # Reset the flag to prevent infinite reloads
        st.session_state.refresh_network = False
        # This will cause network visualization to rebuild completely
        st.empty().button("Refresh Network", key=f"network_refresh_{time.time()}", on_click=lambda: None)
    st.markdown("<h2 class='gradient_text2'>Network Structure</h2>", unsafe_allow_html=True)
    
    # Display simulation information
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {sim_n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type_internal}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a network visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [2, 1]})
    
    # Import networkx for graph visualization
    import networkx as nx
    
    # Make sure adj_matrix is defined for all network types
    if network_type_internal == "All-to-All":
        # For all-to-all, create a fully connected matrix with uniform coupling
        network_adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(network_adj_matrix, 0)  # No self-connections
    elif network_type_internal == "Nearest Neighbor":
        # For nearest neighbor, create a ring topology
        network_adj_matrix = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            # Connect to left and right neighbors on the ring
            network_adj_matrix[i, (i-1) % n_oscillators] = 1
            network_adj_matrix[i, (i+1) % n_oscillators] = 1
    elif network_type_internal == "Random":
        # For random, create random connections with 20% probability
        np.random.seed(random_seed)  # Use same seed for reproducibility
        network_adj_matrix = np.random.random((n_oscillators, n_oscillators)) < 0.2
        network_adj_matrix = network_adj_matrix.astype(float)
        np.fill_diagonal(network_adj_matrix, 0)  # No self-connections
    else:  # Custom Adjacency Matrix
        if adj_matrix is not None:
            # Make sure we're using a copy of the matrix to avoid modifying the original
            network_adj_matrix = np.array(adj_matrix, copy=True)
            
            # This section handles the case where network_adj_matrix dimensions
            # don't match n_oscillators, which might happen in rare cases
            if network_adj_matrix.shape[0] != n_oscillators:
                print(f"Network matrix dimensions mismatch: matrix is {network_adj_matrix.shape} but n_oscillators={n_oscillators}")
                
                # Here we directly use the matrix at its current dimensions without changing session state
                # This is safe because we're not modifying widget values, just local processing
                n_oscillators = network_adj_matrix.shape[0]
                print(f"Using matrix at its native dimension: {n_oscillators}x{n_oscillators}")
            
            # Print detailed debug info
            print(f"Using custom adjacency matrix with shape {network_adj_matrix.shape}")
            print(f"Sum of elements: {np.sum(network_adj_matrix)}")
            print(f"Number of non-zero elements: {np.count_nonzero(network_adj_matrix)}")
            if network_adj_matrix.shape[0] >= 3:
                print(f"Sample (top-left 3x3):")
                print(network_adj_matrix[:3, :3])
        else:
            print("Warning: No custom adjacency matrix provided, using fallback")
            # Create a default matrix but NOT fully connected
            # Use a simple ring structure instead as fallback
            network_adj_matrix = np.zeros((n_oscillators, n_oscillators))
            for i in range(n_oscillators):
                network_adj_matrix[i, (i-1) % n_oscillators] = 1
                network_adj_matrix[i, (i+1) % n_oscillators] = 1
    
    # Create a graph visualization using networkx
    # Ensure the adjacency matrix has no self-loops (diagonal should be zero)
    np.fill_diagonal(network_adj_matrix, 0)
    
    # Print debug info
    if network_type_internal == "Custom Adjacency Matrix":
        print(f"Network adjacency matrix before creating graph:")
        print(f"Shape: {network_adj_matrix.shape}")
        print(f"Sum of elements: {np.sum(network_adj_matrix)}")
        print(f"Number of non-zero elements: {np.count_nonzero(network_adj_matrix)}")
        print(f"Sample of matrix: {network_adj_matrix[:3, :3] if network_adj_matrix.shape[0] >= 3 else network_adj_matrix}")
        
        # Ensure the adjacency matrix has proper values for building a network
        # Some users might enter very small values (like 0.01) that don't register as edges
        # Convert anything > 0.1 to a definite edge to ensure network is visible
        adj_for_network = network_adj_matrix.copy()
        adj_for_network[adj_for_network > 0.1] = 1.0
        if np.count_nonzero(adj_for_network) != np.count_nonzero(network_adj_matrix):
            print(f"Enhancing {np.count_nonzero(adj_for_network) - np.count_nonzero(network_adj_matrix)} weak connections for visualization")
        
        # Use the enhanced matrix for network visualization only
        G = nx.from_numpy_array(adj_for_network)
    else:
        # Use original matrix for standard network types
        G = nx.from_numpy_array(network_adj_matrix)
    
    # Create custom colormap that matches our gradient_text1 theme for nodes
    custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                ["#9933FF", "#FF33FF", "#FFAA00", "#50FF96"], 
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
    if network_type_internal == "Nearest Neighbor":
        # Circular layout for nearest neighbor (ring)
        pos = nx.circular_layout(G)
    # Special case for the Etz Hayim (Tree of Life) matrix - 10x10 matrix with specific structure
    elif network_type_internal == "Custom Adjacency Matrix" and network_adj_matrix.shape[0] == 10 and np.count_nonzero(network_adj_matrix) >= 40:
        # This appears to be our special Etz Hayim matrix - use a custom tree-like layout
        # Create a dictionary with fixed positions for this specific matrix
        # These positions follow a traditional Sephirotic tree arrangement with improved spacing
        fixed_positions = {
            0: (0.0, 1.0),     # Keter (Crown) - top position
            1: (-0.4, 0.8),    # Chokhmah (Wisdom) - upper right
            2: (0.4, 0.8),     # Binah (Understanding) - upper left
            3: (-0.7, 0.4),    # Chesed (Kindness) - middle right
            4: (0.7, 0.4),     # Gevurah (Strength) - middle left
            5: (0.0, 0.4),     # Tiferet (Beauty) - center
            6: (-0.7, 0.0),    # Netzach (Victory) - lower right
            7: (0.7, 0.0),     # Hod (Splendor) - lower left
            8: (0.0, -0.4),    # Yesod (Foundation) - bottom center
            9: (0.0, -0.8)     # Malkhut (Kingdom) - bottom
        }
        pos = fixed_positions
        
        # Add a note that we're using the special layout
        st.info("Using special 'Etz Hayim' (Tree of Life) layout for this adjacency matrix.")
    elif n_oscillators <= 20:
        # Spring layout for smaller networks
        pos = nx.spring_layout(G, seed=random_seed)
    else:
        # Circular layout is better for visualization with many nodes
        pos = nx.circular_layout(G)
    
    # Create graph visualization
    ax1.set_facecolor('#121212')
    
    # Debug information about the graph
    print(f"Graph info: nodes={len(G.nodes())}, edges={len(G.edges())}")
    if len(G.edges()) == 0:
        print("WARNING: Graph has no edges! Check adjacency matrix values.")
        # Force at least some edges for visualization by creating a ring
        temp_G = nx.cycle_graph(network_adj_matrix.shape[0])
        pos = nx.circular_layout(temp_G)
        edges = nx.draw_networkx_edges(temp_G, pos, ax=ax1, alpha=0.7,
                                  edge_color='#ff5500', width=1.5, 
                                  style='dashed')  # Use orange dashed lines to indicate fallback
        # Add warning to graph
        ax1.text(0.5, 0.5, "Warning: No edges detected in custom matrix\nShowing placeholder network",
              horizontalalignment='center', verticalalignment='center',
              transform=ax1.transAxes, color='#ff5500', fontsize=14)
    else:
        # Draw the graph with dark blue edges to match adjacency matrix
        edges = nx.draw_networkx_edges(G, pos, ax=ax1, alpha=0.7, 
                                   edge_color='#0070db', width=1.5)
    
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
    
    # Generate brighter versions of node colors for edges
    bright_node_colors = []
    for color in node_colors:
        # Convert hex to RGB and make it brighter
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        # Make RGB values brighter (closer to white)
        r = min(255, int(r * 1.5))
        g = min(255, int(g * 1.5))
        b = min(255, int(b * 1.5))
        bright_node_colors.append(f"#{r:02x}{g:02x}{b:02x}")
    
    # No glow effect for network nodes to keep the visualization clean
    
    # Draw the main nodes on top of glow effects
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, 
                               node_color=node_colors, 
                               node_size=node_size, alpha=0.9, 
                               edgecolors=bright_node_colors, linewidths=1.0)
    
    # Add node labels only if there are relatively few nodes
    if n_oscillators <= 15:
        labels = {i: str(i) for i in range(n_oscillators)}
        nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, 
                           font_color='white', font_weight='bold')
        
    # Add title and styling
    ax1.set_title(f'Oscillator Network Graph ({network_type_internal})', 
               color='white', fontsize=14, pad=15)
    ax1.set_axis_off()
    
    # Add a legend explaining node colors with brighter outline
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.0), 
                markeredgecolor='#ae56ff', markersize=10, label='Lowest frequency (violet)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.33), 
                markeredgecolor='#ff50ff', markersize=10, label='Lower frequency (magenta)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(0.67), 
                markeredgecolor='#ffc060', markersize=10, label='Higher frequency (orange)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=custom_cmap(1.0), 
                markeredgecolor='#70e898', markersize=10, label='Highest frequency (green)')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', 
            frameon=True, framealpha=0.7, facecolor='#121212', 
            edgecolor='#555555', labelcolor='white')
    
    # Create a heatmap of the adjacency matrix using blue gradient
    # Create a custom blue colormap for the adjacency matrix (swapped light/dark and adjusted intensity)
    blue_cmap = LinearSegmentedColormap.from_list("adj_matrix_blue", 
                                             ["#00c2dd", "#109ae8", "#0070db"], 
                                             N=256)
    # Apply custom colormap - dark blue for 0s, light blue for 1s
    im = ax2.imshow(network_adj_matrix, cmap=blue_cmap)
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
    
    # Special description for Etz Hayim matrix
    if network_type_internal == "Custom Adjacency Matrix" and network_adj_matrix.shape[0] == 10 and np.count_nonzero(network_adj_matrix) >= 40:
        st.markdown("""
        <div class='section'>
            <p>The network visualization shows the <b>Etz Hayim</b> (Tree of Life) configuration:</p>
            <ul>
                <li><b>Left:</b> Graph representing the ten Sephirot (emanations) in the Kabbalistic Tree of Life, arranged in their traditional positions</li>
                <li><b>Right:</b> Adjacency matrix showing the 22 paths connecting the Sephirot</li>
            </ul>
            <p>The nodes represent (from top to bottom, right to left):</p>
            <ol>
                <li>Keter (Crown) - Will and the origin of divine revelation</li>
                <li>Chokhmah (Wisdom) - Beginning of conscious thought</li>
                <li>Binah (Understanding) - Processing and understanding</li>
                <li>Chesed (Kindness) - Expansion, loving kindness</li>
                <li>Gevurah (Strength) - Restriction, judgment and discipline</li>
                <li>Tiferet (Beauty) - Harmony, balance, integration</li>
                <li>Netzach (Victory) - Endurance and overcoming</li>
                <li>Hod (Splendor) - Surrender, sincerity, and gratitude</li>
                <li>Yesod (Foundation) - Connection and bonding force</li>
                <li>Malkhut (Kingdom) - Physical manifestation and action</li>
            </ol>
            <p>This structure offers a fascinating system to study synchronization patterns across interconnected oscillators.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
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
    # Update current tab in session state
    st.session_state.current_tab = "Distributions"
    st.markdown("<h2 class='gradient_text2'>Initial Distributions</h2>", unsafe_allow_html=True)
    
    # Display simulation information at the top
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {sim_n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type_internal}</span>
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
                                                ["#00ffff", "#00aaff"], 
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
    # Update current tab in session state
    st.session_state.current_tab = "Animation"
    st.markdown("<h2 class='gradient_text2'>Interactive Animation</h2>", unsafe_allow_html=True)
    
    # Display simulation information at the top
    st.markdown(f"""
    <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
        <span><b>Oscillators:</b> {sim_n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type_internal}</span>
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
        
        # Add subtle circle rings for reference in white to match unit circle
        for radius in [0.25, 0.5, 0.75]:
            ring = plt.Circle((0, 0), radius, fill=False, color='#ffffff', 
                            linestyle=':', alpha=0.25, zorder=2)
            ax_circle.add_patch(ring)
        
        # Draw unit circle with glow effect - using white
        circle_glow = plt.Circle((0, 0), 1.02, fill=False, color='#ffffff', alpha=0.3, linewidth=3, zorder=3)
        ax_circle.add_patch(circle_glow)
        
        # Main unit circle in white
        circle = plt.Circle((0, 0), 1, fill=False, color='#ffffff', linestyle='-', 
                         linewidth=1.5, alpha=0.8, zorder=4)
        ax_circle.add_patch(circle)
        
        # Plot oscillators
        phases_at_time = phases[:, time_idx]
        x = np.cos(phases_at_time)
        y = np.sin(phases_at_time)
        
        # Create custom colormap that matches our gradient_text1 theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                     ["#8A2BE2", "#FF00FF", "#FFA500", "#50C878"], 
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
        
        # Enhanced scatter plot with oscillator colors and cool effects
        # Create brighter versions of the colors for outlines and glow
        bright_oscillator_colors = np.copy(oscillator_colors)
        for i in range(len(bright_oscillator_colors)):
            # Make RGB values brighter (closer to white) but preserve alpha
            bright_oscillator_colors[i, :3] = np.minimum(1.0, bright_oscillator_colors[i, :3] * 1.7)  # 70% brighter
        
        # First add a larger glow effect for each oscillator
        for i in range(n_oscillators):
            glow = plt.Circle((x[i], y[i]), 0.11, fill=True, 
                          color=oscillator_colors[i], alpha=0.3, zorder=7)
            ax_circle.add_patch(glow)
            
        # Add a subtle pulse effect with secondary glow
        for i in range(n_oscillators):
            second_glow = plt.Circle((x[i], y[i]), 0.14, fill=True, 
                               color=bright_oscillator_colors[i], alpha=0.15, zorder=6)
            ax_circle.add_patch(second_glow)
        
        # Use custom edge colors and increased size for main points
        sc = ax_circle.scatter(x, y, facecolors=oscillator_colors, edgecolors=bright_oscillator_colors, s=200, 
                         alpha=0.9, linewidth=1.5, zorder=10)
        
        # Calculate and show order parameter
        r = order_parameter[time_idx]
        psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
        
        # Draw arrow showing mean field with glow effect - using bright blue color
        # First add glow/shadow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.07, head_length=0.12, fc='#00c2ff', ec='#00c2ff', 
                       width=0.03, alpha=0.3, zorder=5)
        
        # Then add main arrow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.05, head_length=0.1, fc='#00a5ff', ec='#00a5ff', 
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
                                                     ["#8A2BE2", "#FF00FF", "#FFA500", "#50C878"], 
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
                                             ["#00c2dd", "#109ae8", "#0070db"], 
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
    if col1.button("⏪ Previous", use_container_width=True):
        if st.session_state.time_index > 0:
            st.session_state.time_index -= 1
            st.rerun()
    
    # Simplified Play/Pause button
    if col2.button("⏯️ Play/Pause", use_container_width=True):
        # Toggle animation state
        animate = True
        # Let the animation code run
    
    # Next frame button
    if col3.button("⏩ Next", use_container_width=True):
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
        <p>The <b>bottom left plot</b> shows oscillators on a unit circle. Each colored dot represents an oscillator at its current phase position. The blue arrow shows the mean field vector, with length equal to the order parameter r.</p>
        <p>The <b>bottom right plot</b> shows the order parameter over time, with color-coded dots showing the synchronization level from 0 (no synchronization) to 1 (complete synchronization).</p>
        <p>Click "⏯️ Play/Pause" to watch all three visualizations animate together to see the synchronization process in real-time.</p>
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
            
            # JSON EXPORT/IMPORT PANEL
            st.markdown("<h3 class='gradient_text1'>JSON Configuration Export/Import</h3>", unsafe_allow_html=True)
            
            # Create two tabs for export and import
            export_tab, import_tab = st.tabs(["Export Configuration to JSON", "Import Configuration from JSON"])
            
            # Export to JSON section
            with export_tab:
                if not configs:
                    st.info("No saved configurations to export. Save a configuration first.")
                else:
                    # Create a dropdown with configuration names for export
                    export_options = {f"{c['name']} (Oscillators: {c['n_oscillators']}, K: {c['coupling_strength']})": c['id'] for c in configs}
                    export_config = st.selectbox("Select Configuration to Export", options=list(export_options.keys()), key="export_select")
                    
                    json_filename = st.text_input("Filename", value="kuramoto_config.json", key="export_filename")
                    
                    if st.button("Export to JSON", type="primary", key="export_button"):
                        try:
                            export_id = export_options[export_config]
                            file_path = export_configuration_to_json(export_id, json_filename)
                            
                            if file_path:
                                # Read the file content for download
                                with open(file_path, 'rb') as f:
                                    file_content = f.read()
                                
                                # Provide download button
                                st.download_button(
                                    label="Download JSON file",
                                    data=file_content,
                                    file_name=json_filename,
                                    mime="application/json"
                                )
                                
                                st.success(f"Configuration exported successfully to {file_path}")
                            else:
                                st.error("Failed to export configuration to JSON.")
                        except Exception as e:
                            st.error(f"Error exporting configuration: {str(e)}")
            
            # Import from JSON section
            with import_tab:
                uploaded_file = st.file_uploader("Upload JSON Configuration", type=["json"])
                
                if uploaded_file is not None:
                    # Save the uploaded file temporarily
                    temp_filename = f"temp_import_{int(time.time())}.json"
                    with open(temp_filename, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    
                    # Show preview of the configuration
                    try:
                        with open(temp_filename, 'r') as f:
                            config_preview = json.load(f)
                        
                        st.subheader("Configuration Preview")
                        preview_col1, preview_col2 = st.columns(2)
                        
                        preview_data = {
                            "Name": config_preview.get('name', 'Unknown'),
                            "Oscillators": config_preview.get('n_oscillators', 'Unknown'),
                            "Coupling": config_preview.get('coupling_strength', 'Unknown'),
                            "Network Type": config_preview.get('network_type', 'Unknown'),
                            "Frequency Distribution": config_preview.get('frequency_distribution', 'Unknown'),
                            "Simulation Time": config_preview.get('simulation_time', 'Unknown'),
                            "Time Step": config_preview.get('time_step', 'Unknown'),
                            "Random Seed": config_preview.get('random_seed', 'Unknown')
                        }
                        
                        # Display in two columns for better organization
                        for i, (key, value) in enumerate(preview_data.items()):
                            if i < len(preview_data) // 2:
                                preview_col1.write(f"**{key}:** {value}")
                            else:
                                preview_col2.write(f"**{key}:** {value}")
                        
                        st.markdown("---")
                        
                        # Import options
                        save_to_db = st.checkbox("Save to database", value=True, key="import_save_db")
                        custom_name = st.text_input("Configuration name (if saving to database)", 
                                                  value=config_preview.get('name', 'Imported Configuration'),
                                                  key="import_name")
                        
                        if st.button("Import Configuration", type="primary", key="import_button"):
                            try:
                                # Update the name in the file if custom name provided
                                if save_to_db and custom_name != config_preview.get('name', ''):
                                    config_preview['name'] = custom_name
                                    with open(temp_filename, 'w') as f:
                                        json.dump(config_preview, f)
                                
                                # Import the configuration
                                result = import_configuration_from_json(temp_filename, save_to_db=save_to_db)
                                
                                if isinstance(result, int) and save_to_db:
                                    st.success(f"Configuration imported and saved to database with ID: {result}")
                                    # Ask to load the imported configuration
                                    if st.button("Load this configuration now", key="load_imported"):
                                        config = get_configuration(result)
                                        if config:
                                            # Special handling for adjacency matrix when loading from database
                                            if config.get('network_type') == "Custom Adjacency Matrix" and config.get('adjacency_matrix') is not None:
                                                print(f"Loading custom adjacency matrix from database: shape {config['adjacency_matrix'].shape}")
                                                st.session_state.network_type = "Custom Adjacency Matrix"
                                                
                                            st.session_state.loaded_config = config
                                            st.success("Configuration loaded! Applying settings...")
                                            st.rerun()
                                else:
                                    st.success("Configuration imported successfully!")
                                    # Store directly in session state for immediate use
                                    if not save_to_db and st.button("Use this configuration now", key="use_imported"):
                                        # Special handling for adjacency matrix when loading directly
                                        if isinstance(result, dict) and result.get('network_type') == "Custom Adjacency Matrix" and result.get('adjacency_matrix') is not None:
                                            print(f"Loading custom adjacency matrix directly: shape {result['adjacency_matrix'].shape}")
                                            st.session_state.network_type = "Custom Adjacency Matrix"
                                            
                                        st.session_state.loaded_config = result
                                        st.success("Applying imported settings...")
                                        st.rerun()
                            except Exception as e:
                                st.error(f"Error importing configuration: {str(e)}")
                            finally:
                                # Clean up temporary file
                                import os
                                if os.path.exists(temp_filename):
                                    os.remove(temp_filename)
                    except Exception as e:
                        st.error(f"Error parsing JSON file: {str(e)}")