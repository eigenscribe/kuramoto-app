import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Let's strip down all CSS in the app to the minimum needed
st.markdown("""
<style>
    /* No special styling - let Streamlit's defaults handle alignment */
</style>
""", unsafe_allow_html=True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
from src.models.kuramoto_model import KuramotoModel
from src.database.database import save_configuration, get_configuration, list_configurations
import time

# Function to parse JSON parameters input
def parse_json_parameters(json_string):
    """
    Parse a JSON string containing Kuramoto simulation parameters.
    
    Expected format:
    {
        "n_oscillators": 10,
        "coupling_strength": 1.0,
        "network_type": "All-to-All", 
        "simulation_time": 10.0,
        "time_step": 0.1,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.2
        },
        "adjacency_matrix": [[0,1,1,...], [1,0,1,...], ...] (optional)
    }
    
    Returns:
    dict: Parameters dictionary containing all parsed values, or error message
    """
    try:
        # Parse the JSON string
        params = json.loads(json_string)
        
        # Initialize the result dictionary with default values
        result = {
            "n_oscillators": 10,
            "coupling_strength": 1.0,
            "network_type": "All-to-All",
            "simulation_time": 10.0,
            "time_step": 0.1,
            "random_seed": 42,
            "frequency_distribution": "Normal",
            "frequency_parameters": {
                "mean": 0.0,
                "std": 0.2
            },
            "adjacency_matrix": None
        }
        
        # Update with values from the JSON
        if "n_oscillators" in params:
            result["n_oscillators"] = int(params["n_oscillators"])
            
        if "coupling_strength" in params:
            result["coupling_strength"] = float(params["coupling_strength"])
            
        if "network_type" in params:
            valid_types = ["All-to-All", "Nearest Neighbor", "Random", "Custom Adjacency Matrix"]
            if params["network_type"] in valid_types:
                result["network_type"] = params["network_type"]
            
        if "simulation_time" in params:
            result["simulation_time"] = float(params["simulation_time"])
            
        # time_step is now automatically calculated based on oscillator frequencies
        # We keep this for backward compatibility with existing JSON configs
        if "time_step" in params:
            # We acknowledge the parameter but don't use it directly
            pass
            
        if "random_seed" in params:
            result["random_seed"] = int(params["random_seed"])
            
        if "frequency_distribution" in params:
            valid_distributions = ["Normal", "Uniform", "Custom", "Golden Ratio", "Bimodal"]
            if params["frequency_distribution"] in valid_distributions:
                result["frequency_distribution"] = params["frequency_distribution"]
            
        if "frequency_parameters" in params:
            fp = params["frequency_parameters"]
            if "mean" in fp and result["frequency_distribution"] == "Normal":
                result["frequency_parameters"]["mean"] = float(fp["mean"])
            if "std" in fp and result["frequency_distribution"] == "Normal":
                result["frequency_parameters"]["std"] = float(fp["std"])
            if "min" in fp and result["frequency_distribution"] == "Uniform":
                result["frequency_parameters"]["min"] = float(fp["min"])
            if "max" in fp and result["frequency_distribution"] == "Uniform":
                result["frequency_parameters"]["max"] = float(fp["max"])
            if "custom_values" in fp and result["frequency_distribution"] == "Custom":
                # Convert custom frequency values to floats
                result["frequency_parameters"]["custom_values"] = [float(x) for x in fp["custom_values"]]
        
        # Process the adjacency matrix if provided
        if "adjacency_matrix" in params:
            matrix_data = params["adjacency_matrix"]
            if isinstance(matrix_data, list) and len(matrix_data) > 0:
                try:
                    # Convert to numpy array and validate
                    adj_matrix = np.array(matrix_data, dtype=float)
                    
                    # Check if the matrix is square
                    if adj_matrix.shape[0] == adj_matrix.shape[1]:
                        result["adjacency_matrix"] = adj_matrix
                        # If adjacency matrix is provided, force network type to Custom
                        result["network_type"] = "Custom Adjacency Matrix"
                    else:
                        return None, f"Adjacency matrix must be square. Current shape: {adj_matrix.shape}"
                except Exception as e:
                    return None, f"Error processing adjacency matrix: {str(e)}"
        
        return result, None
        
    except Exception as e:
        # Return error message if parsing fails
        return None, str(e)

# Initialize session state for configuration loading
if 'loaded_config' not in st.session_state:
    st.session_state.loaded_config = None

# Check for temp imported parameters and apply them
# This handles imports from the JSON Parameter Import Section
if 'temp_imported_params' in st.session_state:
    params = st.session_state.temp_imported_params
    
    # Update session state with the parsed parameters
    st.session_state.n_oscillators = params["n_oscillators"]
    st.session_state.coupling_strength = params["coupling_strength"]
    st.session_state.network_type = params["network_type"]
    st.session_state.simulation_time = params["simulation_time"]
    # time_step is no longer used, it's automatically calculated
    st.session_state.random_seed = params["random_seed"]
    st.session_state.freq_type = params["frequency_distribution"]
    
    # Update frequency parameters based on distribution type
    if params["frequency_distribution"] == "Normal":
        st.session_state.freq_mean = params["frequency_parameters"]["mean"]
        st.session_state.freq_std = params["frequency_parameters"]["std"]
    elif params["frequency_distribution"] == "Uniform":
        st.session_state.freq_min = params["frequency_parameters"]["min"]
        st.session_state.freq_max = params["frequency_parameters"]["max"]
    elif params["frequency_distribution"] == "Custom" and "custom_values" in params["frequency_parameters"]:
        st.session_state.custom_freqs = ", ".join(str(x) for x in params["frequency_parameters"]["custom_values"])
    
    # Handle custom adjacency matrix if present
    if params["adjacency_matrix"] is not None:
        matrix = params["adjacency_matrix"]
        
        # Convert matrix to string representation for the text area
        matrix_str = ""
        for row in matrix:
            matrix_str += ", ".join(str(val) for val in row) + "\n"
        
        # Update session state for adjacency matrix
        st.session_state.adj_matrix_input = matrix_str.strip()
        st.session_state.loaded_adj_matrix = matrix
    
    # Clear the temp parameters to avoid reapplying
    del st.session_state.temp_imported_params

# Apply loaded configuration if available
if st.session_state.loaded_config is not None:
    config = st.session_state.loaded_config
    
    # Update session state with configuration values
    st.session_state.n_oscillators = config['n_oscillators']
    st.session_state.coupling_strength = config['coupling_strength']
    st.session_state.simulation_time = config['simulation_time']
    # time_step is no longer needed - it's automatically calculated
    st.session_state.random_seed = int(config['random_seed']) # Ensure it's an integer
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

# Page config is now at the top of the file
# This comment is kept to maintain file structure

# Import Aclonica font from Google Fonts
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Aclonica&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Load custom CSS
with open("src/styles/styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Get the base64 encoded image
import base64
with open("static/images/wisp.base64", "r") as f:
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

# Let Streamlit handle the styling
# No custom CSS needed here

# Create main sidebar parameters header at the top
st.sidebar.markdown("<h2 class='gradient_text1'>Simulation Parameters</h2>", unsafe_allow_html=True)

# Add Time Controls Section first
st.sidebar.markdown("<h3 class='gradient_text1'>Time Controls</h3>", unsafe_allow_html=True)

# Simulation time parameters
simulation_time = st.sidebar.slider(
    "Simulation Time",
    min_value=1.0,
    max_value=200.0,
    value=100.0,  # Set default value to 100.0
    step=1.0,
    help="Total simulation time",
    key="simulation_time"
)

# Time step is now automatically calculated based on oscillator frequencies and doesn't need a UI control
# Default value maintained for backward compatibility with database functions
time_step = 0.01

# Initialize model with specified parameters
# Use session state to prevent warnings about duplicate initialization
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42

random_seed = int(st.sidebar.number_input(
    "Random Seed", 
    min_value=0,
    step=1,
    help="Seed for reproducibility",
    key="random_seed"
))


# Add separator before individual parameters
st.sidebar.markdown("<hr style='margin: 15px 0px; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)

# Create subheading for manual parameters
st.sidebar.markdown("<h3 class='gradient_text1'>Manual Configuration</h3>", unsafe_allow_html=True)

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
    st.session_state.simulation_time = 100.0
# Default time_step value maintained for backward compatibility
time_step = 0.01
# random_seed is now initialized directly in the widget section
if 'network_type' not in st.session_state:
    st.session_state.network_type = "Random"
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
    step=1,
    help="Number of oscillators in the system",
    key="n_oscillators"
)

# Coupling strength slider
coupling_strength = st.sidebar.slider(
    "Coupling Strength (K)",
    min_value=0.0,
    max_value=10.0,
    step=0.1,
    help="Strength of coupling between oscillators",
    key="coupling_strength"
)

# Frequency distribution type
freq_type = st.sidebar.selectbox(
    "Frequency Distribution",
    ["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"],
    index=["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"].index(st.session_state.freq_type) if st.session_state.freq_type in ["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"] else 0,
    help="Distribution of natural frequencies",
    key="freq_type"
)

# Parameters for frequency distribution
if freq_type == "Normal":
    freq_mean = st.sidebar.slider("Mean", -2.0, 2.0, step=0.1, key="freq_mean")
    freq_std = st.sidebar.slider("Standard Deviation", 0.1, 3.0, step=0.1, key="freq_std")
    frequencies = np.random.normal(freq_mean, freq_std, n_oscillators)
    
elif freq_type == "Uniform":
    freq_min = st.sidebar.slider("Minimum", -5.0, 0.0, step=0.1, key="freq_min")
    freq_max = st.sidebar.slider("Maximum", 0.0, 5.0, step=0.1, key="freq_max")
    frequencies = np.random.uniform(freq_min, freq_max, n_oscillators)
    
elif freq_type == "Bimodal":
    peak1 = st.sidebar.slider("Peak 1", -5.0, 0.0, step=0.1, key="peak1")
    peak2 = st.sidebar.slider("Peak 2", 0.0, 5.0, step=0.1, key="peak2")
    mix = np.random.choice([0, 1], size=n_oscillators)
    freq1 = np.random.normal(peak1, 0.3, n_oscillators)
    freq2 = np.random.normal(peak2, 0.3, n_oscillators)
    frequencies = mix * freq1 + (1 - mix) * freq2

elif freq_type == "Golden Ratio":
    # The golden ratio (phi) ≈ 1.618033988749895
    phi = (1 + 5**0.5) / 2
    
    # Create a golden ratio sequence starting at -3
    golden_ratio_start = -3.0
    st.sidebar.markdown(f"""
    <div style="background-color: rgba(255,200,0,0.15); padding: 10px; border-radius: 5px;">
    <p><b>Golden Ratio Distribution</b></p>
    <p>This creates a sequence where each frequency follows the golden ratio (φ ≈ 1.618), 
    starting from {golden_ratio_start}.</p>
    <p>Each oscillator's frequency is: {golden_ratio_start} + i·φ</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Generate frequencies that follow the golden ratio in sequence
    frequencies = np.array([golden_ratio_start + i * phi for i in range(n_oscillators)])
    
else:  # Custom
    custom_freqs = st.sidebar.text_area(
        "Enter custom frequencies (comma-separated)",
        value=st.session_state.custom_freqs,
        height=150,
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

# Time controls moved to the top of the sidebar and random seed moved in-line with them

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
        height=200,
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
                                time_step=0.01,  # Default value for backward compatibility
                                random_seed=random_seed,  # Use the UI's random_seed
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


# Add JSON Configuration section at the bottom of the sidebar
st.sidebar.markdown("<hr style='margin: 15px 0px; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
st.sidebar.markdown("<h3 class='gradient_text1'>JSON Configuration</h3>", unsafe_allow_html=True)

# Initialize session state for JSON example if not present
if 'json_example' not in st.session_state:
    st.session_state.json_example = ""

# Display text area for JSON input (larger and left-aligned)
json_input = st.sidebar.text_area(
    "Import/Export Parameters",
    value=st.session_state.json_example,
    height=200,
    placeholder='Paste your JSON configuration here...',
    help="Enter a valid JSON configuration for the Kuramoto simulation"
)

# Add a collapsible section with examples but without parameter details
with st.sidebar.expander("Examples", expanded=False):
    example_json = {
        "n_oscillators": 10,
        "coupling_strength": 1.0,
        "network_type": "All-to-All", 
        "simulation_time": 100.0,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.2
        }
    }
    
    st.code(json.dumps(example_json, indent=2), language="json")
    
    # Add small-world network example
    st.markdown("**Small-world network example:**")
    
    # Generate a sample small-world network
    n = 10
    sample_matrix = np.zeros((n, n))
    for i in range(n):
        # Connect to neighbors
        for j in range(1, 3):
            sample_matrix[i, (i+j) % n] = 1
            sample_matrix[i, (i-j) % n] = 1
            
    # Add a few random long-range connections
    np.random.seed(42)
    for _ in range(5):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j and sample_matrix[i, j] == 0:
            sample_matrix[i, j] = 1
            sample_matrix[j, i] = 1
            
    # Create example with matrix
    complex_example = {
        "n_oscillators": n,
        "coupling_strength": 0.8,
        "network_type": "Custom Adjacency Matrix",
        "simulation_time": 100.0,
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.1
        },
        "adjacency_matrix": sample_matrix.tolist()
    }
    
    # Add button to use this example - with smaller text
    if st.button("Use Small-World", key="small_world_btn"):
        st.session_state.json_example = json.dumps(complex_example, indent=2)
        st.rerun()

# Add import button and logic
if st.sidebar.button("Import Parameters", key="sidebar_import_json_button"):
    if json_input.strip():
        try:
            # Parse the JSON input
            params, error = parse_json_parameters(json_input)
            
            if error:
                st.sidebar.error(f"Error parsing JSON: {error}")
            else:
                # Update session state with the parsed parameters
                if params is not None:
                    # Store all parameters in a temporary variable in session state
                    # This is to avoid the error when trying to change widget values after initialization
                    st.session_state.temp_imported_params = params
                    
                    # Show success message
                    st.sidebar.success("Parameters imported successfully! Applying settings...")
                    
                    # Rerun the app to apply the changes
                    st.rerun()
                else:
                    st.sidebar.error("Failed to parse JSON parameters. Please check your input format.")
        except Exception as e:
            st.sidebar.error(f"Error processing parameters: {str(e)}")
    else:
        st.sidebar.warning("Please enter JSON configuration before importing.")


# Create tabs for different visualizations (Network is default tab)
tab1, tab2, tab3 = st.tabs(["Network", "Distributions", "Animation"])

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
def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step=None, random_seed=None, 
                  adjacency_matrix=None):
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
    time_step : float, optional
        DEPRECATED: This parameter is no longer used. The time step is now automatically calculated
        based on oscillator frequencies to ensure numerical stability and accuracy.
        The parameter is kept for backward compatibility with saved configurations.
    random_seed : int, optional
        Seed for random number generation
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix defining network connectivity
        
    Returns:
    --------
    tuple
        (model, times, phases, order_parameter)
    """
    # Convert random_seed to integer to prevent type errors
    if random_seed is not None:
        random_seed = int(random_seed)
    
    # Initialize the model with given parameters
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        adjacency_matrix=adjacency_matrix,
        simulation_time=simulation_time,
        random_seed=random_seed
    )
    
    # Run the simulation with automatically calculated time step
    try:
        # Force at least 500 time points for visualization
        times, phases, order_parameter = model.simulate(min_time_points=500)
        print("Simulation successful!")
        print(f"  times shape: {times.shape if hasattr(times, 'shape') else 'N/A'}")
        print(f"  phases shape: {phases.shape if hasattr(phases, 'shape') else 'N/A'}")
        print(f"  order_parameter shape: {order_parameter.shape if hasattr(order_parameter, 'shape') else 'N/A'}")
    except Exception as e:
        print(f"Error in simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        # Return empty arrays for graceful failure
        times = np.linspace(0, simulation_time, 100)
        phases = np.zeros((n_oscillators, 100))
        order_parameter = np.zeros(100)
    
    # Return results
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

# Get current random seed from session state
current_random_seed = st.session_state.random_seed if "random_seed" in st.session_state else 42

# Run simulation
model, times, phases, order_parameter = run_simulation(
    n_oscillators=sim_n_oscillators,
    coupling_strength=coupling_strength,
    frequencies=frequencies,
    simulation_time=simulation_time,
    time_step=None,  # Not actually used, time_step is auto-calculated
    random_seed=current_random_seed,
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
        np.random.seed(current_random_seed)  # Use same seed for reproducibility
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
        pos = nx.spring_layout(G, seed=current_random_seed)
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
    # Adjusted node size to be moderately sized - not too big, not too small
    node_size = max(150, int(800 * (1 / (0.12 * n_nodes + 0.6))))  # Balanced node size
    
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
    # Using binary data without text annotations
    im = ax2.imshow(network_adj_matrix, cmap=blue_cmap, interpolation='nearest')
    plt.colorbar(im, ax=ax2, label='Connection Strength')
    
    # Remove the actual 0/1 text annotations by turning off tick labels
    ax2.set_xticks([])
    ax2.set_yticks([])
    
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
            <p>Figure Description:</p>
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
            <p>Figure Description:</p>
            <ul>
                <li><b>Left:</b> Graph representation of oscillator connections, with nodes colored by natural frequency</li>
                <li><b>Right:</b> Adjacency matrix representation, where each cell (i,j) represents the connection strength between oscillators</li>
            </ul>
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
    
    # Define a consistent figure size for both histograms
    hist_figsize = (4.0, 3.0)
    
    with dist_col1:
        st.markdown("<h4 class='gradient_text2'>Natural Frequency Distribution</h4>", unsafe_allow_html=True)
        
        # Create frequency distribution histogram
        fig_freq, ax_freq = plt.subplots(figsize=hist_figsize)
        
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
        
        # Create initial phase distribution histogram with matching size
        fig_init_phase, ax_init_phase = plt.subplots(figsize=hist_figsize)
        
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
        
        # Always add mean phase marker with red vertical line
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
    
    # Safety check to prevent index out of bounds errors
    max_valid_index = len(times) - 1
    if time_index > max_valid_index:
        # Reset the time index to a valid value
        time_index = max_valid_index
        st.session_state.time_index = max_valid_index
        st.warning(f"Time index was reset to maximum value ({max_valid_index})")
    
    # Initialize animation variables
    animate = False
    animation_speed = 3.0
    
    # Get time and order parameter values for the current time index (now with bounds checking)
    current_time = times[time_index]
    current_r = order_parameter[time_index]
    
    # Import needed module
    from matplotlib.collections import LineCollection
    
    # Function to create the phase visualization
    def create_phase_plot(time_idx):
        # Add bounds checking to prevent index errors
        max_valid_idx = min(time_idx, len(times) - 1)
        
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
        
        # Plot oscillators - with bounds checking
        # Make sure we're using a valid time index
        safe_time_idx = max_valid_idx
        
        # Safety check for phase data dimensions
        if safe_time_idx < phases.shape[1]:
            phases_at_time = phases[:, safe_time_idx]
            x = np.cos(phases_at_time)
            y = np.sin(phases_at_time)
        else:
            # Fallback to initial phases if time index is beyond data
            initial_phases = phases[:, 0]
            x = np.cos(initial_phases)
            y = np.sin(initial_phases)
            # Show a warning in the plot
            ax_circle.text(0, 0, "Invalid time index", 
                         ha='center', va='center', color='red', fontsize=12)
        
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
        
        # Enhanced scatter plot with oscillator colors - with reduced glow effect
        # Create slightly brighter versions of the colors for outlines and glow
        bright_oscillator_colors = np.copy(oscillator_colors)
        for i in range(len(bright_oscillator_colors)):
            # Make RGB values brighter (closer to white) but preserve alpha - reduced from 1.7 to 1.3
            bright_oscillator_colors[i, :3] = np.minimum(1.0, bright_oscillator_colors[i, :3] * 1.3)  # 30% brighter
        
        # First add a smaller glow effect for each oscillator
        for i in range(n_oscillators):
            # Reduced size from 0.11 to 0.07 and alpha from 0.3 to 0.2
            glow = plt.Circle((x[i], y[i]), 0.07, fill=True, 
                          color=oscillator_colors[i], alpha=0.2, zorder=7)
            ax_circle.add_patch(glow)
            
        # Add a subtle pulse effect with smaller secondary glow (or remove if too much)
        for i in range(n_oscillators):
            # Reduced size from 0.14 to 0.09 and alpha from 0.15 to 0.1
            second_glow = plt.Circle((x[i], y[i]), 0.09, fill=True, 
                               color=bright_oscillator_colors[i], alpha=0.1, zorder=6)
            ax_circle.add_patch(second_glow)
        
        # Use custom edge colors and increased size for main points - slightly smaller
        sc = ax_circle.scatter(x, y, facecolors=oscillator_colors, edgecolors=bright_oscillator_colors, s=180, 
                         alpha=0.9, linewidth=1.3, zorder=10)
        
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
        # Add bounds checking to prevent index errors
        max_valid_idx = min(time_idx, len(times) - 1)
        
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
        
        # Use the safety bounds check variable
        safe_time_idx = max_valid_idx
        
        # Plot all oscillators as dots up to the current time point
        for i in range(n_oscillators):
            color = oscillator_colors[i]
            
            # Create a brighter version of the color for edge
            # Extract RGB values and make them brighter
            rgb = matplotlib.colors.to_rgb(color)
            # Create brighter version (closer to white)
            bright_color = tuple(min(1.0, c * 1.5) for c in rgb)
            
            # Plot oscillator phases as filled dots with color gradient
            ax.scatter(times[:safe_time_idx+1], phases[i, :safe_time_idx+1] % (2 * np.pi), 
                     facecolors=color, edgecolor=bright_color, alpha=0.7, s=50, 
                     linewidth=0.5, zorder=5)
            
            # Add a subtle connecting line with low opacity
            ax.plot(times[:safe_time_idx+1], phases[i, :safe_time_idx+1] % (2 * np.pi), 
                  color=color, alpha=0.2, linewidth=0.8, zorder=2)
            
            # Highlight current position with a larger filled marker
            ax.scatter([times[safe_time_idx]], [phases[i, safe_time_idx] % (2 * np.pi)], 
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
        ax.set_title(f'Oscillator Phases at t={times[safe_time_idx]:.2f}', 
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
        # Add bounds checking to prevent index errors
        max_valid_idx = min(time_idx, len(times) - 1)
        
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
        
        # Use the safe index for all references
        safe_time_idx = max_valid_idx
        
        # Plot order parameter with filled gradient dots and brighter outline
        base_colors = [cmap(r) for r in order_parameter[:safe_time_idx+1]]
        edge_colors = []
        
        # Create brighter versions of each color for the outlines
        for color in base_colors:
            rgb = matplotlib.colors.to_rgb(color)
            # Make RGB values brighter but preserve alpha
            bright_color = tuple(min(1.0, c * 1.5) for c in rgb)
            edge_colors.append(bright_color)
        
        scatter = ax.scatter(times[:safe_time_idx+1], order_parameter[:safe_time_idx+1],
                          facecolors=base_colors, edgecolors=edge_colors,
                          s=70, alpha=0.9, linewidth=0.5, zorder=10)
        
        # Removed connecting line as requested
        
        # Highlight current position with a larger filled marker
        if safe_time_idx > 0:
            # Get color and make brighter version for outline
            current_color = cmap(order_parameter[safe_time_idx])
            rgb = matplotlib.colors.to_rgb(current_color)
            bright_current = tuple(min(1.0, c * 1.5) for c in rgb)
            
            ax.scatter([times[safe_time_idx]], [order_parameter[safe_time_idx]], 
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
        ax.set_title(f'Phase Synchronization at t={times[safe_time_idx]:.2f}', 
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
    
    # Put animation controls first (at the top)
    # Create centered columns for control buttons
    col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
    
    # Create a placeholder for displaying time step info 
    current_time_placeholder = col4.empty()
    
    # Function to update time step display
    def update_time_step_display(time_idx):
        # Calculate the actual time step (difference between consecutive time points)
        if time_idx > 0:
            time_step = times[time_idx] - times[time_idx-1]
        else:
            # Use first difference for the first point
            if len(times) > 1:
                time_step = times[1] - times[0]
            else:
                time_step = 0
                
        current_time = times[time_idx]
        current_percent = (time_idx / (len(times) - 1)) * 100
        
        # Update the placeholder with the time step information
        current_time_placeholder.markdown(f"""
        <div style="padding: 10px; border-radius: 5px; background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(255, 0, 255, 0.2)); 
                    border: 1px solid rgba(255, 255, 255, 0.1); text-align: center;">
            <span style="font-size: 0.85rem; color: white;">Δt = {time_step:.5f}</span><br>
            <span style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.7);">t = {current_time:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Set a fixed animation speed value
    animation_speed = 3.0  # Fixed moderate animation speed
    
    # Previous frame button (in second column for centering)
    if col2.button("⏪ Previous", use_container_width=True):
        if st.session_state.time_index > 0:
            st.session_state.time_index -= 1
            st.rerun()
    
    # Simplified Play/Pause button with text inside (centered in middle column)
    play_button_text = "⏯️ Play"
    if col3.button(play_button_text, use_container_width=True):
        # Toggle animation state
        animate = True
        # Let the animation code run
    
    # Next frame button 
    if col4.button("⏩ Next", use_container_width=True):
        if st.session_state.time_index < len(times) - 1:
            st.session_state.time_index += 1
            st.rerun()
    
    # Initial display of time step
    update_time_step_display(st.session_state.time_index)
    
    # Add a slider to manually control visualization time point AFTER the buttons
    playback_container = st.container()
    time_index = playback_container.slider(
        "Time Point", 
        min_value=0, 
        max_value=len(times)-1, 
        value=st.session_state.time_index,
        help="Manually select a specific time point to display"
    )
    
    # Update session state when slider is moved
    if st.session_state.time_index != time_index:
        st.session_state.time_index = time_index
            
    # If animation is triggered
    if animate:
        # Get the current time index as the starting point
        start_idx = st.session_state.time_index
        
        # Ensure we're not starting beyond the available data
        max_valid_index = len(times) - 1
        if start_idx > max_valid_index:
            start_idx = 0
            st.session_state.time_index = 0
        
        # Calculate how many frames to skip based on speed
        frame_skip = max(1, int(11 - animation_speed))
        
        # Set up a progress bar
        progress_bar = st.progress(0)
        
        # Animation loop with bounds checking
        for i in range(start_idx, min(len(times), max_valid_index + 1), frame_skip):
            # Update the session state
            st.session_state.time_index = i
            
            # Update progress bar - with safeguards to prevent division by zero
            denominator = max(1, max_valid_index - start_idx)  # Ensure non-zero denominator
            progress = min(1.0, (i - start_idx) / denominator)
            progress_bar.progress(progress)
            
            # Update all three plots
            # Safety checks for index values
            plot_idx = min(i, max_valid_index)
            circle_plot_placeholder.pyplot(create_phase_plot(plot_idx))
            phases_plot_placeholder.pyplot(create_oscillator_phases_plot(plot_idx))
            order_plot_placeholder.pyplot(create_order_parameter_plot(plot_idx))
            
            # Update the time step display with the current time index
            update_time_step_display(plot_idx)
            
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
        <p>Click "⏯️ Play" to watch all three visualizations animate together to see the synchronization process in real-time.</p>
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

# Database and configurations sections removed