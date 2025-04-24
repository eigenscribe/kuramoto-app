import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path for proper imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal modules
from src.models.kuramoto_model import KuramotoModel
from src.database.database import (
    store_simulation, get_simulation, list_simulations, delete_simulation,
    save_configuration, list_configurations, get_configuration, delete_configuration,
    export_configuration_to_json, import_configuration_from_json
)
from src.utils.ml_helper import analyze_simulation_data

# Import external packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
import time
import os
import networkx as nx
from scipy.optimize import minimize_scalar
from matplotlib.collections import LineCollection
import tempfile
import datetime

# Load CSS from file
def load_css(css_file):
    with open(css_file, "r") as f:
        return f.read()

# Apply CSS styles
try:
    css_content = load_css('src/styles/main.css')
except FileNotFoundError:
    # Fallback for direct development
    try:
        css_content = load_css('styles.css')
    except FileNotFoundError:
        css_content = ""
        st.error("Could not load CSS file")

# Get the base64 encoded image for background
try:
    with open("static/images/wisp.base64", "r") as f:
        encoded_image = f.read()
except FileNotFoundError:
    # Fallback path
    try: 
        with open("wisp.base64", "r") as f:
            encoded_image = f.read()
    except FileNotFoundError:
        encoded_image = ""
        st.warning("Background image not found")

# Apply both CSS and background image
st.markdown(f"""
<style>
{css_content}

@import url('https://fonts.googleapis.com/css2?family=Aclonica&display=swap');

.stApp {{
    background-image: url("data:image/jpg;base64,{encoded_image}");
    background-size: cover;
    background-repeat: no-repeat;
    font-family: 'Aclonica', sans-serif;
}}
</style>
""", unsafe_allow_html=True)

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
        "auto_optimize": true, (optional)
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
        params = json.loads(json_string)
        
        # Required parameters
        required_params = ["n_oscillators", "coupling_strength", "network_type", 
                          "simulation_time", "time_step", "random_seed", 
                          "frequency_distribution"]
        
        for param in required_params:
            if param not in params:
                return {"error": f"Missing required parameter: {param}"}
        
        # Validate frequency distribution and parameters
        if params["frequency_distribution"].lower() == "normal":
            if "frequency_parameters" not in params or not isinstance(params["frequency_parameters"], dict):
                return {"error": "Missing or invalid frequency_parameters for Normal distribution"}
            if "mean" not in params["frequency_parameters"] or "std" not in params["frequency_parameters"]:
                return {"error": "Normal distribution requires 'mean' and 'std' parameters"}
                
        elif params["frequency_distribution"].lower() == "uniform":
            if "frequency_parameters" not in params or not isinstance(params["frequency_parameters"], dict):
                return {"error": "Missing or invalid frequency_parameters for Uniform distribution"}
            if "min" not in params["frequency_parameters"] or "max" not in params["frequency_parameters"]:
                return {"error": "Uniform distribution requires 'min' and 'max' parameters"}
                
        elif params["frequency_distribution"].lower() == "golden":
            # No additional parameters needed for Golden Ratio distribution
            pass
            
        else:
            return {"error": f"Unsupported frequency distribution: {params['frequency_distribution']}"}
        
        # Validate network type and adjacency matrix
        if params["network_type"].lower() == "custom":
            if "adjacency_matrix" not in params or not isinstance(params["adjacency_matrix"], list):
                return {"error": "Custom network type requires an adjacency_matrix parameter"}
            
            # Validate adjacency matrix dimensions
            n = params["n_oscillators"]
            if len(params["adjacency_matrix"]) != n:
                return {"error": f"Adjacency matrix rows ({len(params['adjacency_matrix'])}) don't match n_oscillators ({n})"}
            
            for row in params["adjacency_matrix"]:
                if len(row) != n:
                    return {"error": f"Adjacency matrix columns ({len(row)}) don't match n_oscillators ({n})"}
        
        # Standardize network type to lowercase
        params["network_type"] = params["network_type"].lower()
        
        # Standardize frequency distribution to lowercase
        params["frequency_distribution"] = params["frequency_distribution"].lower()
        
        return params
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON format: {str(e)}"}
    except Exception as e:
        return {"error": f"Error parsing parameters: {str(e)}"}

# Function to run Kuramoto simulation with given parameters
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
    # Initialize the model
    model = KuramotoModel(n_oscillators=n_oscillators, 
                         coupling_strength=coupling_strength,
                         frequencies=frequencies,
                         simulation_time=simulation_time,
                         time_step=time_step,
                         random_seed=random_seed,
                         adjacency_matrix=adjacency_matrix)
    
    # Optimize time step if requested
    optimized_time_step = None
    if auto_optimize:
        optimized_time_step = model.optimize_time_step(frequencies, coupling_strength, adjacency_matrix,
                                                      safety_factor=safety_factor)
        # Update the model with the optimized time step
        model.time_step = optimized_time_step
    
    # Run the simulation
    times, phases = model.run_simulation()
    
    # Calculate the order parameter
    r, psi = model.calculate_order_parameter(phases)
    order_parameter = {"r": r, "psi": psi}
    
    return model, times, phases, order_parameter, optimized_time_step

# Title and header
st.title("Kuramoto Model Simulator")
st.markdown("""
This simulator explores the dynamics of the Kuramoto model, which describes synchronization 
phenomena in systems of coupled oscillators.
""")

# Main layout with tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🔄 Simulation", 
    "🌐 Network Visualization", 
    "📊 Analysis",
    "💾 Save/Load"
])

# Sidebar with parameters
st.sidebar.markdown("<h2 class='gradient_text1'>Simulation Parameters</h2>", unsafe_allow_html=True)

# Time parameters with green gradient styling
st.sidebar.markdown("<h3 class='gradient_text1'>Time Controls</h3>", unsafe_allow_html=True)
simulation_time = st.sidebar.slider(
    "Simulation Time", 
    min_value=1.0, 
    max_value=100.0, 
    value=20.0, 
    help="Total time to simulate (arbitrary units)"
)
time_step = st.sidebar.slider(
    "Time Step", 
    min_value=0.01, 
    max_value=0.5, 
    value=0.1, 
    step=0.01,
    help="Time step for numerical integration"
)
auto_optimize = st.sidebar.checkbox(
    "Auto-optimize Time Step", 
    value=False,
    help="Automatically select an optimal time step based on system parameters"
)

if auto_optimize:
    safety_factor = st.sidebar.slider(
        "Safety Factor", 
        min_value=0.1, 
        max_value=0.99, 
        value=0.8,
        step=0.05, 
        help="Lower values are more stable but slower"
    )
else:
    safety_factor = 0.8

# Create parameter sections using different gradient colors
st.sidebar.markdown("<h3 class='gradient_text3'>Oscillator Parameters</h3>", unsafe_allow_html=True)
n_oscillators = st.sidebar.slider(
    "Number of Oscillators", 
    min_value=2, 
    max_value=100, 
    value=10,
    help="Number of oscillators in the system"
)

coupling_strength = st.sidebar.slider(
    "Coupling Strength (K)", 
    min_value=0.0, 
    max_value=5.0, 
    value=1.0, 
    step=0.1,
    help="Strength of coupling between oscillators"
)

# Network section with blue gradient
st.sidebar.markdown("<h3 class='gradient_text2'>Network Connectivity</h3>", unsafe_allow_html=True)
network_type = st.sidebar.selectbox(
    "Network Type",
    options=["All-to-All", "Ring", "Random", "Custom"],
    index=0,
    help="Topology of the oscillator network"
)

if network_type == "Random":
    connection_probability = st.sidebar.slider(
        "Connection Probability", 
        min_value=0.1, 
        max_value=1.0, 
        value=0.5, 
        step=0.1,
        help="Probability of connection between any two oscillators"
    )

if network_type == "Custom":
    st.sidebar.markdown("Define custom adjacency matrix in the Network tab")

# Frequency distribution with another gradient
st.sidebar.markdown("<h3 class='gradient_text2'>Frequency Distribution</h3>", unsafe_allow_html=True)
freq_distribution = st.sidebar.selectbox(
    "Distribution Type",
    options=["Normal", "Uniform", "Golden"],
    index=0,
    help="Distribution of natural frequencies among oscillators"
)

if freq_distribution == "Normal":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        freq_mean = st.number_input("Mean", value=0.0, step=0.1, format="%.2f")
    with col2:
        freq_std = st.number_input("Std Dev", value=0.2, step=0.1, min_value=0.01, format="%.2f")
elif freq_distribution == "Uniform":
    col1, col2 = st.sidebar.columns(2)
    with col1:
        freq_min = st.number_input("Min", value=-0.5, step=0.1, format="%.2f")
    with col2:
        freq_max = st.number_input("Max", value=0.5, step=0.1, format="%.2f")
        
# Randomization with gradient
st.sidebar.markdown("<h3 class='gradient_text3'>Randomization</h3>", unsafe_allow_html=True)
random_seed = st.sidebar.number_input(
    "Random Seed", 
    value=42, 
    min_value=0, 
    step=1,
    help="Seed for reproducible randomization"
)

# Button to run simulation with gradient style
run_button = st.sidebar.button("▶️ Run Simulation", type="primary", use_container_width=True)

# Custom adjacency matrix input in Network tab
with tab2:
    st.subheader("Network Connectivity Visualization")
    
    if network_type == "Custom":
        st.markdown("""
        Define your custom adjacency matrix below. 
        - Enter a space-separated matrix with 0s and 1s
        - Each row should be on a new line
        - A 1 indicates a connection between oscillators
        - Make sure dimensions match the number of oscillators
        """)
        
        # Set up a default matrix if needed
        if 'custom_adjacency_matrix' not in st.session_state or st.session_state.custom_adjacency_matrix is None:
            default_matrix = "\n".join([" ".join(["1" if i != j else "0" for j in range(n_oscillators)]) for i in range(n_oscillators)])
            st.session_state.custom_adjacency_matrix = default_matrix
            
        # Create a larger text area for the adjacency matrix
        adjacency_str = st.text_area(
            "Adjacency Matrix",
            value=st.session_state.custom_adjacency_matrix,
            height=200,
            key="adjacency_matrix_input"
        )
        
        # Parse the adjacency matrix from the string input
        try:
            adjacency_list = [[int(val) for val in row.split()] for row in adjacency_str.strip().split('\n')]
            adjacency_matrix = np.array(adjacency_list)
            
            # Validate dimensions
            if adjacency_matrix.shape != (n_oscillators, n_oscillators):
                st.error(f"Matrix dimensions ({adjacency_matrix.shape}) don't match number of oscillators ({n_oscillators})")
                adjacency_matrix = None
                
            # Update session state
            st.session_state.custom_adjacency_matrix = adjacency_str
        except Exception as e:
            st.error(f"Error parsing adjacency matrix: {str(e)}")
            adjacency_matrix = None
    else:
        adjacency_matrix = None  # Will be generated by model based on network_type

# Generate natural frequencies based on distribution
if freq_distribution == "Normal":
    np.random.seed(random_seed)
    frequencies = np.random.normal(freq_mean, freq_std, n_oscillators)
elif freq_distribution == "Uniform":
    np.random.seed(random_seed)
    frequencies = np.random.uniform(freq_min, freq_max, n_oscillators)
elif freq_distribution == "Golden":
    # Golden ratio distribution: natural frequencies follow the golden ratio (approx 1.618)
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    frequencies = np.array([-3 + i * phi for i in range(n_oscillators)])
    frequencies -= np.mean(frequencies)  # Center around zero

# Run simulation when button is clicked
if run_button:
    with st.spinner("Running simulation..."):
        # Convert network type for the model
        if network_type == "All-to-All":
            model_network_type = "all-to-all"
        elif network_type == "Ring":
            model_network_type = "ring"
        elif network_type == "Random":
            model_network_type = "random"
            # Pass connection probability parameter
            if 'connection_probability' in locals():
                adjacency_matrix = np.random.rand(n_oscillators, n_oscillators) < connection_probability
                np.fill_diagonal(adjacency_matrix, 0)  # No self-connections
                # Make symmetric
                adjacency_matrix = np.logical_or(adjacency_matrix, adjacency_matrix.T).astype(int)
        
        # Run the simulation
        current_random_seed = random_seed
        model, times, phases, order_parameter, optimized_time_step = run_simulation(
            n_oscillators=n_oscillators,
            coupling_strength=coupling_strength,
            frequencies=frequencies,
            simulation_time=simulation_time,
            time_step=time_step,
            random_seed=current_random_seed,
            adjacency_matrix=adjacency_matrix,
            auto_optimize=auto_optimize,
            safety_factor=safety_factor
        )
        
        # Display optimized time step info if requested
        if auto_optimize and optimized_time_step is not None:
            st.sidebar.success(f"Optimized time step: {optimized_time_step:.5f} (originally {time_step:.5f})")
            
        # Store the simulation in the database
        config_name = f"Simulation {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        config_data = {
            "n_oscillators": n_oscillators,
            "coupling_strength": coupling_strength,
            "network_type": network_type.lower(),
            "simulation_time": simulation_time,
            "time_step": time_step if not auto_optimize else optimized_time_step,
            "auto_optimize": auto_optimize,
            "random_seed": current_random_seed,
            "frequency_distribution": freq_distribution.lower(),
            "frequency_parameters": {
                "mean": freq_mean if freq_distribution == "Normal" else None,
                "std": freq_std if freq_distribution == "Normal" else None,
                "min": freq_min if freq_distribution == "Uniform" else None,
                "max": freq_max if freq_distribution == "Uniform" else None
            }
        }
        
        # Store the results
        simulation_data = {
            "times": times.tolist(),
            "phases": phases.tolist(),
            "order_parameter": {
                "r": order_parameter["r"].tolist(),
                "psi": order_parameter["psi"].tolist()
            },
            "frequencies": frequencies.tolist(),
            "adjacency_matrix": adjacency_matrix.tolist() if adjacency_matrix is not None else None
        }
        
        # Store in database
        sim_id = store_simulation(config_data, simulation_data)
        save_configuration(config_name, config_data)
        
        st.success(f"Simulation completed and saved (ID: {sim_id})")
        
        # Store for visualization
        st.session_state.last_simulation = {
            "times": times,
            "phases": phases,
            "order_parameter": order_parameter,
            "frequencies": frequencies,
            "adjacency_matrix": adjacency_matrix,
            "config": config_data
        }