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
import time

# Updated import paths to reflect the new structure
try:
    # Try importing with src prefix (for when run from root)
    from src.models.kuramoto_model import KuramotoModel
    from src.database.database import store_simulation, get_simulation, list_simulations  # Import necessary database functions
except ImportError:
    # Fall back to direct import (for when run directly)
    from models.kuramoto_model import KuramotoModel
    from database.database import store_simulation, get_simulation, list_simulations  # Import necessary database functions

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
        
        # Check required fields
        required_fields = ["n_oscillators", "coupling_strength", "network_type", 
                           "simulation_time", "time_step", "random_seed", 
                           "frequency_distribution"]
        
        for field in required_fields:
            if field not in params:
                return {"error": f"Missing required field: {field}"}
        
        # Check frequency parameters
        if "frequency_parameters" not in params:
            return {"error": "Missing frequency_parameters object"}
            
        if params["frequency_distribution"] == "Normal":
            if "mean" not in params["frequency_parameters"] or "std" not in params["frequency_parameters"]:
                return {"error": "Normal distribution requires mean and std parameters"}
        elif params["frequency_distribution"] == "Uniform":
            if "min" not in params["frequency_parameters"] or "max" not in params["frequency_parameters"]:
                return {"error": "Uniform distribution requires min and max parameters"}
        elif params["frequency_distribution"] == "Custom":
            if "values" not in params["frequency_parameters"]:
                return {"error": "Custom frequency distribution requires values array"}
            if len(params["frequency_parameters"]["values"]) != params["n_oscillators"]:
                return {"error": "Custom frequency values length must match n_oscillators"}
        elif params["frequency_distribution"] == "Golden Ratio":
            # No extra params needed for golden ratio
            pass
        else:
            return {"error": f"Unknown frequency distribution: {params['frequency_distribution']}"}
            
        # Check adjacency matrix if provided
        if "adjacency_matrix" in params:
            # Must be square and match n_oscillators
            if len(params["adjacency_matrix"]) != params["n_oscillators"]:
                return {"error": "Adjacency matrix rows must match n_oscillators"}
            for row in params["adjacency_matrix"]:
                if len(row) != params["n_oscillators"]:
                    return {"error": "Adjacency matrix must be square"}
        
        # All checks passed
        return params
        
    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON: {str(e)}"}
    except Exception as e:
        return {"error": f"Error parsing parameters: {str(e)}"}


# Function to generate network adjacency matrix
def generate_network_matrix(network_type, n_oscillators, random_seed=42):
    """
    Generate an adjacency matrix for the oscillator network.
    
    Parameters:
    -----------
    network_type : str
        Type of network ('all-to-all', 'random', 'ring', 'small-world')
    n_oscillators : int
        Number of oscillators
    random_seed : int, optional
        Seed for random number generation
        
    Returns:
    --------
    ndarray
        Adjacency matrix of shape (n_oscillators, n_oscillators)
    """
    np.random.seed(random_seed)
    
    if network_type == "All-to-All":
        # All-to-all: fully connected network
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        # Remove self-connections (diagonal elements)
        np.fill_diagonal(adj_matrix, 0)
        
    elif network_type == "Random":
        # Random network with connection probability 0.5
        adj_matrix = np.random.random((n_oscillators, n_oscillators)) < 0.5
        # Make symmetric and remove self-connections
        adj_matrix = np.logical_or(adj_matrix, adj_matrix.T).astype(float)
        np.fill_diagonal(adj_matrix, 0)
        
    elif network_type == "Ring":
        # Ring network: each oscillator connected to its 2 nearest neighbors
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        for i in range(n_oscillators):
            adj_matrix[i, (i-1) % n_oscillators] = 1
            adj_matrix[i, (i+1) % n_oscillators] = 1
            
    elif network_type == "Small-World":
        # Small-world network: start with ring, then rewire some connections
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        # Start with ring
        for i in range(n_oscillators):
            adj_matrix[i, (i-1) % n_oscillators] = 1
            adj_matrix[i, (i+1) % n_oscillators] = 1
        
        # Rewire with probability 0.2
        rewire_prob = 0.2
        for i in range(n_oscillators):
            for j in range(n_oscillators):
                if adj_matrix[i, j] == 1 and np.random.random() < rewire_prob:
                    # Remove this connection
                    adj_matrix[i, j] = 0
                    adj_matrix[j, i] = 0  # Keep symmetric
                    
                    # Add a new random connection
                    new_target = np.random.randint(0, n_oscillators)
                    while new_target == i or adj_matrix[i, new_target] == 1:
                        new_target = np.random.randint(0, n_oscillators)
                    
                    adj_matrix[i, new_target] = 1
                    adj_matrix[new_target, i] = 1  # Keep symmetric
    
    elif network_type == "Scale-Free":
        # Implementation of Barabási–Albert model for scale-free networks
        # Start with a small complete graph
        m0 = min(5, n_oscillators)  # Initial network size
        m = min(2, m0)  # Number of edges to attach from a new node to existing nodes
        
        adj_matrix = np.zeros((n_oscillators, n_oscillators))
        
        # Create initial complete graph
        for i in range(m0):
            for j in range(i+1, m0):
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Add remaining nodes with preferential attachment
        for i in range(m0, n_oscillators):
            # Calculate attachment probabilities based on degree
            degrees = np.sum(adj_matrix[:i], axis=1)
            probs = degrees / degrees.sum() if degrees.sum() > 0 else None
            
            # Connect to m existing nodes
            targets = np.random.choice(i, size=m, replace=False, p=probs)
            for target in targets:
                adj_matrix[i, target] = 1
                adj_matrix[target, i] = 1
                
    elif network_type == "Custom Adjacency Matrix":
        # Return an identity matrix as a placeholder - will be replaced by custom input
        adj_matrix = np.eye(n_oscillators)
        
    elif network_type == "Tree of Life":
        # Specialized Kabbalistic Tree of Life network structure
        # Only valid for 10 oscillators (10 Sephirot)
        if n_oscillators != 10:
            # Fall back to all-to-all if not 10 oscillators
            adj_matrix = np.ones((n_oscillators, n_oscillators))
            np.fill_diagonal(adj_matrix, 0)
        else:
            # Create the Tree of Life structure (Etz Hayim)
            # 10 Sephirot with 22 connecting paths
            adj_matrix = np.zeros((10, 10))
            
            # Define the 22 paths of the Tree of Life
            paths = [
                # Pillar of Mercy
                (0, 1), (1, 3), (3, 6), (6, 8),
                # Pillar of Severity 
                (0, 2), (2, 4), (4, 7), (7, 9),
                # Middle Pillar
                (0, 5), (5, 8), (8, 9),
                # Horizontal paths
                (1, 2), (1, 5), (2, 5), (3, 4), (3, 5), (4, 5), (6, 7), (6, 5), (7, 5),
                # Diagonal paths
                (1, 4), (2, 3)
            ]
            
            # Set connections based on paths
            for i, j in paths:
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1  # Symmetric
    
    else:
        # Default to all-to-all
        adj_matrix = np.ones((n_oscillators, n_oscillators))
        np.fill_diagonal(adj_matrix, 0)
        
    return adj_matrix


# Function to generate natural frequencies for oscillators
def generate_frequencies(distribution_type, n_oscillators, params, random_seed=42):
    """
    Generate natural frequencies for oscillators.
    
    Parameters:
    -----------
    distribution_type : str
        Type of frequency distribution ('normal', 'uniform', 'custom')
    n_oscillators : int
        Number of oscillators
    params : dict
        Parameters for the distribution (e.g., mean and std for normal)
    random_seed : int, optional
        Seed for random number generation
        
    Returns:
    --------
    ndarray
        Array of natural frequencies
    """
    np.random.seed(random_seed)
    
    if distribution_type == "Normal":
        mean = params.get("mean", 0.0)
        std = params.get("std", 0.2)
        frequencies = np.random.normal(mean, std, n_oscillators)
        
    elif distribution_type == "Uniform":
        min_freq = params.get("min", -0.5)
        max_freq = params.get("max", 0.5)
        frequencies = np.random.uniform(min_freq, max_freq, n_oscillators)
        
    elif distribution_type == "Custom":
        # Custom frequencies provided directly
        frequencies = np.array(params.get("values", np.zeros(n_oscillators)))
        
    elif distribution_type == "Golden Ratio":
        # Generate frequencies following the Golden Ratio pattern
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio ≈ 1.618
        frequencies = np.array([-3 + (i * phi) % 6 for i in range(n_oscillators)])
        
    else:
        # Default to normal distribution
        frequencies = np.random.normal(0, 0.2, n_oscillators)
        
    return frequencies


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
    
    def estimate_optimal_time_step(max_freq, max_coupling, n_oscillators, safety_factor=0.8):
        """Estimate optimal time step based on system parameters"""
        # Consider both frequency and coupling dynamics
        critical_time = min(
            1.0 / (2 * max_freq),  # Based on Nyquist for highest frequency
            1.0 / (max_coupling * n_oscillators)  # Based on coupling strength
        )
        # Apply safety factor
        return critical_time * safety_factor
    
    # Optimize time step if requested
    optimized_time_step = None
    if auto_optimize:
        max_freq = max(abs(np.max(frequencies)), abs(np.min(frequencies)))
        max_coupling = coupling_strength
        if adjacency_matrix is not None:
            # Consider network structure in optimization
            max_degree = np.max(np.sum(adjacency_matrix, axis=1))
            max_coupling *= max_degree / n_oscillators
        
        # Estimate optimal time step
        optimized_time_step = estimate_optimal_time_step(
            max_freq, max_coupling, n_oscillators, safety_factor
        )
        
        # Use the optimized time step
        time_step = min(time_step, optimized_time_step)
    
    # Create the model
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        natural_frequencies=frequencies,
        adjacency_matrix=adjacency_matrix,
        random_seed=random_seed
    )
    
    # Run the simulation
    times = np.arange(0, simulation_time, time_step)
    phases = model.simulate(times)
    
    # Calculate order parameter r over time
    order_parameter = model.calculate_order_parameter(phases)
    
    return model, times, phases, order_parameter, optimized_time_step


# Function to generate NetworkX graph from adjacency matrix
def generate_network_graph(adjacency_matrix, positions=None):
    """
    Generate a NetworkX graph from adjacency matrix and optionally with
    pre-defined node positions.
    
    Parameters:
    -----------
    adjacency_matrix : ndarray
        Adjacency matrix defining the network
    positions : dict, optional
        Dictionary of node positions {node_id: (x, y)}
        
    Returns:
    --------
    tuple
        (graph, positions)
    """
    import networkx as nx
    
    # Create graph from adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    
    # Generate positions if not provided
    if positions is None:
        if adjacency_matrix.shape[0] == 10 and adjacency_matrix[0, 5] == 1:
            # Check if this is likely the Tree of Life pattern (10 nodes with specific connections)
            # Use predefined Tree of Life layout
            positions = {
                0: (0, 1),       # Keter (Crown)
                1: (-0.5, 0.6),  # Chokhmah (Wisdom)
                2: (0.5, 0.6),   # Binah (Understanding)
                3: (-0.5, 0.2),  # Chesed (Kindness)
                4: (0.5, 0.2),   # Gevurah (Strength)
                5: (0, 0),       # Tiferet (Beauty)
                6: (-0.5, -0.4), # Netzach (Victory)
                7: (0.5, -0.4),  # Hod (Splendor)
                8: (0, -0.8),    # Yesod (Foundation)
                9: (0, -1)       # Malkhut (Kingdom)
            }
        else:
            # Use spring layout for other networks
            positions = nx.spring_layout(G, seed=42)
    
    return G, positions


# Custom colors for network nodes based on frequencies
def get_frequency_color(freq, min_freq, max_freq):
    """
    Generate a color for a node based on its natural frequency.
    
    Parameters:
    -----------
    freq : float
        Natural frequency of the oscillator
    min_freq : float
        Minimum frequency in the network
    max_freq : float
        Maximum frequency in the network
        
    Returns:
    --------
    str
        Hex color code
    """
    # Normalize frequency to [0, 1]
    freq_range = max_freq - min_freq
    if freq_range == 0:
        normalized = 0.5  # If all frequencies are the same
    else:
        normalized = (freq - min_freq) / freq_range
    
    # Create a complex gradient through the color spectrum
    if normalized < 0.25:
        # Violet to magenta gradient
        r = int(128 + (128 * normalized * 4))
        g = int(0)
        b = int(255)
    elif normalized < 0.5:
        # Magenta to orange gradient
        r = int(255)
        g = int(0 + (128 * (normalized - 0.25) * 4))
        b = int(255 - (255 * (normalized - 0.25) * 4))
    elif normalized < 0.75:
        # Orange to yellow-green gradient
        r = int(255 - (128 * (normalized - 0.5) * 4))
        g = int(128 + (127 * (normalized - 0.5) * 4))
        b = int(0)
    else:
        # Yellow-green to green gradient
        r = int(127 - (127 * (normalized - 0.75) * 4))
        g = int(255)
        b = int(0 + (128 * (normalized - 0.75) * 4))
    
    # Convert RGB to hex
    return f"#{r:02x}{g:02x}{b:02x}"


# --------------------------------------------
# Load CSS styling
# --------------------------------------------
def load_css():
    try:
        # Try first with src prefix (when run from root)
        with open("src/styles/main.css") as f:
            css = f.read()
    except FileNotFoundError:
        try:
            # Then try with relative path (when run from src directory)
            with open("styles/main.css") as f:
                css = f.read()
        except FileNotFoundError:
            # Fall back to original path
            with open("styles.css") as f:
                css = f.read()
    
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load background image
def load_background():
    import base64
    try:
        # Try first with static directory (when run from root)
        with open("static/images/wisp.base64", "r") as f:
            background_base64 = f.read()
    except FileNotFoundError:
        try:
            # Try with relative path to static (when run from src directory)
            with open("../static/images/wisp.base64", "r") as f:
                background_base64 = f.read()
        except FileNotFoundError:
            # Fall back to original path
            with open("wisp.base64", "r") as f:
                background_base64 = f.read()
                
    return background_base64


# ----------------------
# Main application code
# ----------------------
def main():
    # Load CSS and background
    load_css()
    background_base64 = load_background()
    
    # Inject background image
    st.markdown(f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{background_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """, unsafe_allow_html=True)

    # --------------------------------------------
    # Set up page title and sidebar
    # --------------------------------------------
    st.markdown("<h1 class='gradient_text'>Kuramoto Synchronization Simulator</h1>", unsafe_allow_html=True)
    
    # Initialize session state for current tab if not already set
    if 'current_tab' not in st.session_state:
        st.session_state.current_tab = "Network"
        
    # Initialize session state for animation time index if not already set
    if 'time_index' not in st.session_state:
        st.session_state.time_index = 0
        
    # Initialize session state for simulation data if not already set
    if 'simulation_data' not in st.session_state:
        st.session_state.simulation_data = None

    # --------------------------------------------
    # SIDEBAR SECTION
    # --------------------------------------------
    with st.sidebar:
        st.markdown("<h3 class='gradient_text'>Simulation Parameters</h3>", unsafe_allow_html=True)
        
        # Oscillator parameters (orange gradient)
        st.markdown("<div class='parameter-section orange-gradient'>", unsafe_allow_html=True)
        st.markdown("<h4>Oscillator Parameters</h4>", unsafe_allow_html=True)
        
        n_oscillators = st.slider("Number of Oscillators", min_value=3, max_value=100, value=10)
        coupling_strength = st.slider("Coupling Strength (K)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Network parameters (blue gradient)
        st.markdown("<div class='parameter-section blue-gradient'>", unsafe_allow_html=True)
        st.markdown("<h4>Network Connectivity</h4>", unsafe_allow_html=True)
        
        network_type = st.selectbox(
            "Network Type", 
            options=["All-to-All", "Ring", "Random", "Small-World", "Scale-Free", "Tree of Life", "Custom Adjacency Matrix"]
        )
        
        custom_matrix = None
        
        if network_type == "Custom Adjacency Matrix":
            st.markdown("Enter adjacency matrix as JSON array of arrays:")
            custom_matrix_str = st.text_area(
                "Adjacency Matrix", 
                value="""[
    [0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0],
    [1, 1, 0, 0, 1],
    [0, 1, 0, 0, 1],
    [0, 0, 1, 1, 0]
]""", 
                height=200
            )
            
            try:
                custom_matrix = json.loads(custom_matrix_str)
                # Validate custom matrix
                if len(custom_matrix) != n_oscillators:
                    st.warning(f"Matrix has {len(custom_matrix)} rows but should have {n_oscillators} for the current number of oscillators")
                for row in custom_matrix:
                    if len(row) != n_oscillators:
                        st.warning(f"Matrix has a row with {len(row)} columns but should have {n_oscillators}")
            except Exception as e:
                st.error(f"Invalid matrix format: {str(e)}")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Frequency distribution parameters (blue gradient)
        st.markdown("<div class='parameter-section blue-gradient'>", unsafe_allow_html=True)
        st.markdown("<h4>Frequency Distribution</h4>", unsafe_allow_html=True)
        
        freq_type = st.selectbox(
            "Distribution Type", 
            options=["Normal", "Uniform", "Golden Ratio", "Custom"]
        )
        
        custom_freqs = None
        
        if freq_type == "Normal":
            freq_mean = st.slider("Mean Frequency", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
            freq_std = st.slider("Standard Deviation", min_value=0.01, max_value=2.0, value=0.2, step=0.01)
        elif freq_type == "Uniform":
            freq_min = st.slider("Minimum Frequency", min_value=-5.0, max_value=0.0, value=-0.5, step=0.1)
            freq_max = st.slider("Maximum Frequency", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
        elif freq_type == "Custom":
            custom_freqs_str = st.text_area(
                "Custom Frequencies", 
                value="[0.0, 0.2, -0.2, 0.4, -0.4]", 
                height=100
            )
            
            try:
                custom_freqs = json.loads(custom_freqs_str)
                # Validate custom frequencies
                if len(custom_freqs) != n_oscillators:
                    st.warning(f"You specified {len(custom_freqs)} frequencies but need {n_oscillators}")
            except Exception as e:
                st.error(f"Invalid frequency format: {str(e)}")
                
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Time parameters (green gradient)
        st.markdown("<div class='parameter-section green-gradient'>", unsafe_allow_html=True)
        st.markdown("<h4>Simulation Time</h4>", unsafe_allow_html=True)
        
        simulation_time = st.slider("Simulation Time", min_value=1.0, max_value=50.0, value=10.0, step=0.5)
        time_step = st.slider("Time Step", min_value=0.01, max_value=0.5, value=0.1, step=0.01)
        
        auto_optimize = st.checkbox("Auto-optimize Time Step", value=True, 
                                  help="Automatically select an optimal time step based on system parameters")
        
        if auto_optimize:
            safety_factor = st.slider("Safety Factor", min_value=0.1, max_value=1.0, value=0.8, step=0.1,
                                    help="Lower values give more conservative (smaller) time steps")
        else:
            safety_factor = 0.8  # Default
            
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Other parameters
        st.markdown("<div class='parameter-section'>", unsafe_allow_html=True)
        st.markdown("<h4>Other Parameters</h4>", unsafe_allow_html=True)
        
        random_seed = st.number_input("Random Seed", min_value=1, max_value=10000, value=42)
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Configure JSON parameters
        with st.expander("JSON Parameter Configuration"):
            st.markdown("""
            Load or save simulation parameters in JSON format. 
            This allows you to store complex configurations and share them.
            """)
            
            json_input = st.text_area(
                "JSON Parameters", 
                height=250,
                value="""{
    "n_oscillators": 10,
    "coupling_strength": 1.0,
    "network_type": "All-to-All",
    "simulation_time": 10.0,
    "time_step": 0.1,
    "random_seed": 42,
    "auto_optimize": true,
    "frequency_distribution": "Normal",
    "frequency_parameters": {
        "mean": 0.0,
        "std": 0.2
    }
}"""
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Load Parameters"):
                    params = parse_json_parameters(json_input)
                    
                    if "error" in params:
                        st.error(params["error"])
                    else:
                        st.success("Parameters loaded successfully!")
                        
                        # Update all widgets with values from JSON
                        st.session_state.n_oscillators = params["n_oscillators"]
                        st.session_state.coupling_strength = params["coupling_strength"]
                        st.session_state.network_type = params["network_type"]
                        st.session_state.simulation_time = params["simulation_time"]
                        st.session_state.time_step = params["time_step"]
                        st.session_state.random_seed = params["random_seed"]
                        st.session_state.auto_optimize = params.get("auto_optimize", True)
                        st.session_state.freq_type = params["frequency_distribution"]
                        
                        # Handle frequency-specific parameters
                        freq_params = params["frequency_parameters"]
                        if params["frequency_distribution"] == "Normal":
                            st.session_state.freq_mean = freq_params["mean"]
                            st.session_state.freq_std = freq_params["std"]
                        elif params["frequency_distribution"] == "Uniform":
                            st.session_state.freq_min = freq_params["min"]
                            st.session_state.freq_max = freq_params["max"]
                        
                        # Handle custom adjacency matrix if provided
                        if "adjacency_matrix" in params:
                            st.session_state.custom_matrix_str = json.dumps(params["adjacency_matrix"], indent=4)
                        
                        # Rerun to update widget values
                        st.experimental_rerun()
            
            with col2:
                if st.button("Generate JSON"):
                    # Create JSON from current parameters
                    params = {
                        "n_oscillators": n_oscillators,
                        "coupling_strength": coupling_strength,
                        "network_type": network_type,
                        "simulation_time": simulation_time,
                        "time_step": time_step,
                        "random_seed": random_seed,
                        "auto_optimize": auto_optimize,
                        "frequency_distribution": freq_type,
                        "frequency_parameters": {}
                    }
                    
                    # Add frequency-specific parameters
                    if freq_type == "Normal":
                        params["frequency_parameters"] = {
                            "mean": freq_mean,
                            "std": freq_std
                        }
                    elif freq_type == "Uniform":
                        params["frequency_parameters"] = {
                            "min": freq_min,
                            "max": freq_max
                        }
                    elif freq_type == "Custom":
                        params["frequency_parameters"] = {
                            "values": custom_freqs
                        }
                    else:
                        params["frequency_parameters"] = {}
                    
                    # Add custom adjacency matrix if provided
                    if custom_matrix is not None:
                        params["adjacency_matrix"] = custom_matrix
                    
                    # Update the JSON text area
                    json_output = json.dumps(params, indent=4)
                    st.session_state.json_input = json_output
                    st.text_area("Generated JSON", value=json_output, height=250)
                    
                    # Option to save to file
                    st.download_button(
                        "Download JSON", 
                        json_output, 
                        file_name="kuramoto_config.json", 
                        mime="application/json"
                    )
            
            # JSON file uploader
            uploaded_file = st.file_uploader("Upload Configuration File", type=["json"])
            if uploaded_file is not None:
                try:
                    # Load JSON from uploaded file
                    params = json.load(uploaded_file)
                    # Display in the text area
                    st.session_state.json_input = json.dumps(params, indent=4)
                    st.success("File loaded successfully! Click 'Load Parameters' to apply.")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
        
        # Run simulation button
        if st.button("🚀 Run Simulation", use_container_width=True):
            with st.spinner("Running simulation..."):
                # Generate adjacency matrix based on network type
                if network_type == "Custom Adjacency Matrix" and custom_matrix is not None:
                    network_adj_matrix = np.array(custom_matrix)
                else:
                    network_adj_matrix = generate_network_matrix(network_type, n_oscillators, random_seed)
                
                # Generate frequencies based on distribution type
                if freq_type == "Normal":
                    freq_params = {"mean": freq_mean, "std": freq_std}
                elif freq_type == "Uniform":
                    freq_params = {"min": freq_min, "max": freq_max}
                elif freq_type == "Custom" and custom_freqs is not None:
                    freq_params = {"values": custom_freqs}
                else:
                    freq_params = {}
                
                frequencies = generate_frequencies(freq_type, n_oscillators, freq_params, random_seed)
                
                # Run the simulation
                model, times, phases, order_parameter, opt_time_step = run_simulation(
                    n_oscillators=n_oscillators,
                    coupling_strength=coupling_strength,
                    frequencies=frequencies,
                    simulation_time=simulation_time,
                    time_step=time_step,
                    random_seed=random_seed,
                    adjacency_matrix=network_adj_matrix,
                    auto_optimize=auto_optimize,
                    safety_factor=safety_factor
                )
                
                # Store the results in session state
                st.session_state.simulation_data = {
                    "model": model,
                    "times": times,
                    "phases": phases,
                    "order_parameter": order_parameter,
                    "frequencies": frequencies,
                    "network_adj_matrix": network_adj_matrix,
                    "sim_n_oscillators": n_oscillators,
                    "coupling_strength": coupling_strength,
                    "random_seed": random_seed,
                    "simulation_time": simulation_time,
                    "time_step": opt_time_step if opt_time_step is not None else time_step,
                    "network_type": network_type,
                    "freq_type": freq_type
                }
                
                # Reset time index for animations
                st.session_state.time_index = 0
                
                st.success(f"Simulation completed! {len(times)} time steps calculated.")
                
                # Display optimization info if used
                if auto_optimize and opt_time_step is not None:
                    if opt_time_step < time_step:
                        st.info(f"Time step optimized: {time_step} → {opt_time_step:.4f}")
                    else:
                        st.info(f"Original time step ({time_step}) was already optimal.")
    
    # --------------------------------------------
    # MAIN CONTENT AREA
    # --------------------------------------------
    
    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["Network", "Distributions", "Animation", "Order Parameter"])
    
    # Check if simulation has been run
    if st.session_state.simulation_data is None:
        for tab in [tab1, tab2, tab3, tab4]:
            with tab:
                st.info("Run a simulation to visualize results.")
        st.stop()  # Stop execution until a simulation is run
    
    # Extract simulation data from session state
    model = st.session_state.simulation_data["model"]
    times = st.session_state.simulation_data["times"]
    phases = st.session_state.simulation_data["phases"]
    order_parameter = st.session_state.simulation_data["order_parameter"]
    frequencies = st.session_state.simulation_data["frequencies"]
    network_adj_matrix = st.session_state.simulation_data["network_adj_matrix"]
    sim_n_oscillators = st.session_state.simulation_data["sim_n_oscillators"]
    coupling_strength = st.session_state.simulation_data["coupling_strength"]
    network_type_internal = st.session_state.simulation_data["network_type"]
    freq_type = st.session_state.simulation_data["freq_type"]

    ########################
    # TAB 1: NETWORK TAB
    ########################
    with tab1:
        # Update current tab in session state
        st.session_state.current_tab = "Network"
        st.markdown("<h2 class='gradient_text1'>Network Structure</h2>", unsafe_allow_html=True)
        
        # Display simulation information at the top
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
            <span><b>Oscillators:</b> {sim_n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type_internal}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Display network visualization
        st.markdown("<h3 class='gradient_text1'>Network Visualization</h3>", unsafe_allow_html=True)
        
        # Generate Matplotlib figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Create network graph
        G, pos = generate_network_graph(network_adj_matrix)
        
        # Log information for debugging
        print(f"Graph info: nodes={len(G.nodes)}, edges={len(G.edges)}")
        
        # Node colors based on natural frequencies
        min_freq = np.min(frequencies)
        max_freq = np.max(frequencies)
        node_colors = [get_frequency_color(freq, min_freq, max_freq) for freq in frequencies]
        
        # Draw edges with appropriate styling
        import networkx as nx
        import matplotlib.collections as mcol
        
        # Set background color for the graph
        ax1.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#121212')
        
        # Create a custom node colormap for more visually appealing colors
        custom_cmap = LinearSegmentedColormap.from_list("node_colormap", 
                                                  ["#8800ff", "#ff00ff", "#ff8800", "#88ff00"], 
                                                  N=256)
        
        # Draw the edges
        edges = nx.draw_networkx_edges(G, pos, ax=ax1, 
                                   edge_color='#555555', alpha=0.7, 
                                   width=1.0)
        
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
        
        # Create animation controls
        st.markdown("<h3 class='gradient_text3'>Animation Controls</h3>", unsafe_allow_html=True)
        
        # Display the current time information
        current_time_idx = st.session_state.time_index
        if current_time_idx >= len(times):
            # Ensure time index is within bounds
            current_time_idx = len(times) - 1
            st.session_state.time_index = current_time_idx
            
        current_time = times[current_time_idx]
        
        # Calculate current order parameter value for display
        current_r = order_parameter[current_time_idx]
            
        # Time display and slider
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.25); padding: 8px; border-radius: 5px; margin-bottom: 10px;'>
            <span style='font-weight: bold;'>Current Time: {current_time:.2f}</span> | 
            <span>Order Parameter: {current_r:.4f}</span>
        </div>
        """, unsafe_allow_html=True)
        
        time_slider = st.slider("Time", min_value=0, max_value=len(times)-1, value=current_time_idx, format="t = %d")
        if time_slider != current_time_idx:
            st.session_state.time_index = time_slider
            st.experimental_rerun()
            
        # Animation control buttons
        anim_col1, anim_col2, anim_col3, anim_col4, anim_col5 = st.columns([1, 1, 1, 1, 2])
        
        with anim_col1:
            if st.button("⏮️ Start", use_container_width=True, help="Go to the beginning"):
                st.session_state.time_index = 0
                st.experimental_rerun()
                
        with anim_col2:
            if st.button("⏪ Back", use_container_width=True, help="Go back 10 steps"):
                st.session_state.time_index = max(0, current_time_idx - 10)
                st.experimental_rerun()
                
        with anim_col3:
            if st.button("⏩ Forward", use_container_width=True, help="Go forward 10 steps"):
                st.session_state.time_index = min(len(times) - 1, current_time_idx + 10)
                st.experimental_rerun()
                
        with anim_col4:
            if st.button("⏭️ End", use_container_width=True, help="Go to the end"):
                st.session_state.time_index = len(times) - 1
                st.experimental_rerun()
                
        with anim_col5:
            # Auto-animation toggle using session state
            if 'auto_animate' not in st.session_state:
                st.session_state.auto_animate = False
                
            if st.button("🔄 Auto Play/Pause", use_container_width=True, help="Automatically advance through time steps"):
                st.session_state.auto_animate = not st.session_state.auto_animate
                
            # If auto-animation is on, advance the time index
            if st.session_state.auto_animate:
                st.markdown("<div style='text-align: center;'><span style='color: #4CAF50;'>Auto Play: ON</span></div>", unsafe_allow_html=True)
                # Automatically advance time index on rerun
                new_idx = (current_time_idx + 1) % len(times)
                # Use JavaScript to rerun after a delay
                st.markdown(f"""
                <script>
                    setTimeout(function(){{
                        window.streamlitRerun();
                    }}, 100);
                </script>
                """, unsafe_allow_html=True)
                # Update time index for next run
                st.session_state.time_index = new_idx
            else:
                st.markdown("<div style='text-align: center;'><span style='color: #F44336;'>Auto Play: OFF</span></div>", unsafe_allow_html=True)
        
        # Animation figures
        anim_col1, anim_col2 = st.columns([1.2, 1.0])
        
        with anim_col1:
            st.markdown("<h4 class='gradient_text3'>Phase Animation</h4>", unsafe_allow_html=True)
            
            # Get current phases for the selected time
            phases_at_time = phases[:, current_time_idx]
            
            # Create phase plot function
            def create_phase_plot(time_idx):
                """Create a phase plot for the specified time index"""
                # Make sure time index is within bounds
                time_idx = min(time_idx, len(times) - 1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(7, 6), subplot_kw={'projection': 'polar'})
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#121212')
                
                # Get phases at the specified time
                phases_at_time = phases[:, time_idx]
                
                # Calculate the order parameter
                r = order_parameter[time_idx]
                psi = np.angle(np.mean(np.exp(1j * phases_at_time)))
                
                # Plot the unit circle
                circle = plt.Circle((0, 0), 1, transform=ax.transData._b, color="#333333", fill=False, linewidth=1)
                ax.add_artist(circle)
                
                # Create a radial grid with a blue glow effect
                for radius in np.linspace(0.2, 1.0, 5):
                    # Add grid circles
                    circle = plt.Circle((0, 0), radius, transform=ax.transData._b, color="#0088aa", fill=False, 
                                    alpha=0.2, linewidth=1)
                    ax.add_artist(circle)
                
                # Add angular grid lines with a blue glow
                for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
                    # Calculate endpoints
                    x = [0, np.cos(angle)]
                    y = [0, np.sin(angle)]
                    # Add line
                    ax.plot(x, y, color="#0088aa", alpha=0.2, linewidth=1, transform=ax.transData._b)
                
                # Zoom in to show oscillator positions better
                ax.set_ylim(0, 1.1)
                
                # Calculate marker sizes based on number of oscillators
                if n_oscillators <= 10:
                    marker_size = 150
                    marker_linewidth = 2
                    ghost_size = 20
                elif n_oscillators <= 30:
                    marker_size = 100
                    marker_linewidth = 1.5
                    ghost_size = 15
                else:
                    marker_size = 50
                    marker_linewidth = 1
                    ghost_size = 10
                
                # Add oscillator markers
                scatter = ax.scatter(phases_at_time, [1] * len(phases_at_time), 
                                 c=node_colors, s=marker_size, alpha=0.9, zorder=5,
                                 edgecolors=bright_node_colors, linewidths=marker_linewidth)
                
                # Add ghost markers near the center to show frequencies
                # Normalize frequencies for visualization
                f_min, f_max = min(frequencies), max(frequencies)
                if f_max > f_min:
                    # Normalize to range [0.1, 0.5] for ghost marker distances
                    normalized_frequencies = 0.1 + 0.4 * (frequencies - f_min) / (f_max - f_min)
                else:
                    normalized_frequencies = [0.3] * len(frequencies)
                
                # Add ghost markers
                ghost_scatter = ax.scatter(phases_at_time, normalized_frequencies, 
                                       c=node_colors, s=ghost_size, alpha=0.5, zorder=4)
                
                # Add order parameter vector
                ax.plot([0, psi], [0, r], color='red', linewidth=3, alpha=0.7, zorder=3)
                ax.scatter(psi, r, color='red', s=100, alpha=0.7, zorder=6)
                
                # Add subtle glow to the order parameter
                circle = plt.Circle((psi, r), 0.05, transform=ax.transData._b, color="red", 
                               alpha=0.2, zorder=2)
                ax.add_artist(circle)
                
                # Add mean phase marker on perimeter
                ax.scatter(psi, 1, color='red', s=100, alpha=0.7, marker='s', zorder=6)
                
                # Add information text
                plt.annotate(f'Time: {times[time_idx]:.2f}', xy=(0.05, 0.05), xycoords='figure fraction',
                         color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.4", 
                                                           alpha=0.4, 
                                                           facecolor='#0d0d0d', 
                                                           edgecolor='#333333'))
                
                plt.annotate(f'r = {r:.4f}', xy=(0.05, 0.95), xycoords='figure fraction',
                         color='white', fontsize=10, bbox=dict(boxstyle="round,pad=0.4", 
                                                           alpha=0.4, 
                                                           facecolor='#0d0d0d', 
                                                           edgecolor='#333333'))
                
                # Make gridlines lighter
                ax.grid(color='#333333', alpha=0.3)
                
                # Set title and styling
                ax.set_title(f'Oscillator Phases at t = {times[time_idx]:.2f}', 
                         color='white', fontsize=14, pad=15)
                
                # Make tick labels lighter
                ax.tick_params(axis='both', colors='#999999')
                
                return fig
            
            # Display the phase plot
            st.pyplot(create_phase_plot(current_time_idx))
            
            # Phase plot description
            st.markdown("""
            <div class='section' style='font-size: 0.9em;'>
                <p>This plot shows the phase of each oscillator on the unit circle:</p>
                <ul>
                    <li>Each <b>colored dot</b> on the circle represents an oscillator</li>
                    <li>The <b>red arrow</b> shows the order parameter vector (length = r)</li>
                    <li>The <b>red square</b> on the perimeter marks the mean phase</li>
                    <li>The ghost markers near the center show relative natural frequencies</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        with anim_col2:
            st.markdown("<h4 class='gradient_text3'>Oscillator Phases</h4>", unsafe_allow_html=True)
            
            # Function to create the oscillator phases plot
            def create_oscillator_phases_plot(time_idx):
                """Create plot showing phases of all oscillators over time with current position marker"""
                # Make sure time index is within bounds
                time_idx = min(time_idx, len(times) - 1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(6, 5))
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#121212')
                
                # Plot phases over time for each oscillator
                for i in range(n_oscillators):
                    # Get phases for this oscillator
                    osc_phases = phases[i, :time_idx+1]
                    
                    # Unwrap phases to avoid jumps from 2π to 0
                    unwrapped = np.unwrap(osc_phases)
                    
                    # Plot line with color based on natural frequency
                    ax.plot(times[:time_idx+1], unwrapped, color=node_colors[i], linewidth=1.5, alpha=0.8)
                    
                    # Add current position marker
                    ax.scatter(times[time_idx], unwrapped[-1], color=node_colors[i], 
                           s=50, edgecolor='white', linewidth=1, zorder=10)
                
                # Add current time vertical line
                ax.axvline(x=times[time_idx], color='#ff5555', linestyle='--', linewidth=1, alpha=0.7)
                
                # Add grid
                ax.grid(True, color='#333333', alpha=0.3, linestyle=':')
                
                # Set labels and title
                ax.set_xlabel('Time', color='white', fontsize=12)
                ax.set_ylabel('Phase (unwrapped)', color='white', fontsize=12)
                ax.set_title('Oscillator Phases Over Time', color='white', fontsize=14, pad=15)
                
                # Make tick labels lighter
                ax.tick_params(axis='both', colors='#999999')
                
                # Add legend if there are few oscillators
                if n_oscillators <= 10:
                    handles = [plt.Line2D([0], [0], color=node_colors[i], linewidth=2) 
                               for i in range(n_oscillators)]
                    labels = [f"Osc {i}" for i in range(n_oscillators)]
                    ax.legend(handles, labels, loc='upper left', framealpha=0.7, 
                           facecolor='#121212', edgecolor='#333333', labelcolor='white')
                
                return fig
            
            # Display the oscillator phases plot
            st.pyplot(create_oscillator_phases_plot(current_time_idx))
            
            # Add order parameter plot
            st.markdown("<h4 class='gradient_text3'>Order Parameter</h4>", unsafe_allow_html=True)
            
            def create_order_parameter_plot(time_idx):
                """Create order parameter plot with current position marker"""
                # Make sure time index is within bounds
                time_idx = min(time_idx, len(times) - 1)
                
                # Create figure
                fig, ax = plt.subplots(figsize=(6, 3))
                
                # Set background color
                ax.set_facecolor('#1a1a1a')
                fig.patch.set_facecolor('#121212')
                
                # Plot order parameter over time
                ax.plot(times[:time_idx+1], order_parameter[:time_idx+1], 
                    color='#ff5555', linewidth=2, alpha=0.8)
                
                # Add current position marker
                ax.scatter(times[time_idx], order_parameter[time_idx], 
                       color='#ff5555', s=80, edgecolor='white', linewidth=1, zorder=10)
                
                # Add current time vertical line
                ax.axvline(x=times[time_idx], color='#ff5555', linestyle='--', linewidth=1, alpha=0.7)
                
                # Add grid
                ax.grid(True, color='#333333', alpha=0.3, linestyle=':')
                
                # Set labels and title
                ax.set_xlabel('Time', color='white', fontsize=12)
                ax.set_ylabel('Order Parameter (r)', color='white', fontsize=12)
                ax.set_title('Order Parameter Over Time', color='white', fontsize=14, pad=15)
                
                # Set y-axis limits to [0, 1]
                ax.set_ylim(0, 1.05)
                
                # Make tick labels lighter
                ax.tick_params(axis='both', colors='#999999')
                
                # Add text with current value
                plt.annotate(f'r = {order_parameter[time_idx]:.4f}', 
                         xy=(0.75, 0.15), xycoords='axes fraction',
                         color='white', fontsize=12, 
                         bbox=dict(boxstyle="round,pad=0.4", 
                               alpha=0.6, 
                               facecolor='#0d0d0d', 
                               edgecolor='#333333'))
                
                return fig
            
            # Display the order parameter plot
            st.pyplot(create_order_parameter_plot(current_time_idx))
        
    ########################
    # TAB 4: ORDER PARAMETER TAB
    ########################
    with tab4:
        # Update current tab in session state
        st.session_state.current_tab = "Order Parameter"
        st.markdown("<h2 class='gradient_text3'>Order Parameter Analysis</h2>", unsafe_allow_html=True)
        
        # Display simulation information at the top
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
            <span style='font-size: 1.2em;'><b>Simulation Information</b></span><br>
            <span><b>Oscillators:</b> {sim_n_oscillators} | <b>Coupling Strength:</b> {coupling_strength} | <b>Network Type:</b> {network_type_internal}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Description of order parameter
        st.markdown("""
        <div class='section'>
            <p>The order parameter, denoted by r(t), is a key metric in the Kuramoto model that quantifies the degree of synchronization among oscillators. 
            It ranges from 0 (complete desynchronization) to 1 (perfect synchronization).</p>
            
            <p>The order parameter is defined as:</p>
            <div style='text-align: center; font-size: 1.2em; padding: 10px 0;'>
                r(t)e<sup>iψ(t)</sup> = 1/N ∑<sub>j=1</sub><sup>N</sup> e<sup>iθ<sub>j</sub>(t)</sup>
            </div>
            
            <p>Where:</p>
            <ul>
                <li>r(t) is the magnitude of the order parameter (degree of synchronization)</li>
                <li>ψ(t) is the average phase of the oscillators</li>
                <li>θ<sub>j</sub>(t) is the phase of oscillator j at time t</li>
                <li>N is the total number of oscillators</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        # Order parameter visualization
        st.markdown("<h3 class='gradient_text3'>Order Parameter Evolution</h3>", unsafe_allow_html=True)
        
        # Create a detailed order parameter plot
        fig_order, ax_order = plt.subplots(figsize=(10, 6))
        
        # Set background color
        ax_order.set_facecolor('#1a1a1a')
        fig_order.patch.set_facecolor('#121212')
        
        # Plot line with gradient color
        from matplotlib.collections import LineCollection
        from matplotlib.colors import ListedColormap
        
        # Create points for colored line segments
        points = np.array([times, order_parameter]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        # Create a custom colormap for the line
        gradient_cm = LinearSegmentedColormap.from_list(
            "order_param_cm", ['#ff0066', '#ff3366', '#ff6666', '#ff9966', '#ffcc66'], N=256)
        
        # Create the colored line collection
        lc = LineCollection(segments, cmap=gradient_cm, linewidth=2.5)
        
        # Set color values based on time
        lc.set_array(times[:-1])
        
        # Add the collection to the plot
        line = ax_order.add_collection(lc)
        
        # Plot a semi-transparent fill below the line
        ax_order.fill_between(times, 0, order_parameter, color='#ff6666', alpha=0.15)
        
        # Enhance the plot with additional elements
        
        # Add a horizontal line for the final value
        final_r = order_parameter[-1]
        ax_order.axhline(y=final_r, color='#ffcc66', linestyle='--', linewidth=1.5, alpha=0.7,
                     label=f'Final value: {final_r:.4f}')
        
        # Add a horizontal line for the mean value
        mean_r = np.mean(order_parameter)
        ax_order.axhline(y=mean_r, color='#66ccff', linestyle=':', linewidth=1.5, alpha=0.7,
                     label=f'Mean value: {mean_r:.4f}')
        
        # Calculate and highlight the point where steady-state is approximately reached
        # First, smooth the order parameter to reduce noise
        window_size = max(5, len(order_parameter) // 50)  # Adaptive window size
        if len(order_parameter) > window_size:
            smoothed_r = np.convolve(order_parameter, np.ones(window_size)/window_size, mode='valid')
            
            # Find where the derivative gets close to zero (steady state)
            # Using a simple difference approximation
            if len(smoothed_r) > 1:
                derivatives = np.abs(np.diff(smoothed_r))
                
                # Identify where derivative falls below threshold
                threshold = 0.01 * (np.max(derivatives) - np.min(derivatives)) + np.min(derivatives)
                steady_indices = np.where(derivatives < threshold)[0]
                
                if len(steady_indices) > 0:
                    # Find the first occurrence of steady state
                    first_steady_idx = steady_indices[0] + window_size // 2  # Adjust for smoothing offset
                    first_steady_idx = min(first_steady_idx, len(times) - 1)  # Ensure within bounds
                    
                    # Mark the steady-state point
                    steady_time = times[first_steady_idx]
                    steady_r = order_parameter[first_steady_idx]
                    
                    ax_order.scatter(steady_time, steady_r, s=100, color='#00ff99', 
                                 edgecolor='white', linewidth=1, zorder=10,
                                 label=f'Steady state at t ≈ {steady_time:.2f}')
                    
                    # Add vertical line at steady-state time
                    ax_order.axvline(x=steady_time, color='#00ff99', linestyle='-', linewidth=1, alpha=0.4)
        
        # Add grid
        ax_order.grid(True, color='#333333', alpha=0.3, linestyle=':')
        
        # Set labels and title
        ax_order.set_xlabel('Time', color='white', fontsize=12, fontweight='bold')
        ax_order.set_ylabel('Order Parameter (r)', color='white', fontsize=12, fontweight='bold')
        ax_order.set_title('Evolution of Synchronization Over Time', 
                       color='white', fontsize=16, fontweight='bold', pad=20)
        
        # Set y-axis limits to [0, 1]
        ax_order.set_ylim(-0.05, 1.05)
        
        # Set x-axis limits
        ax_order.set_xlim(0, times[-1] * 1.02)
        
        # Make tick labels lighter
        ax_order.tick_params(axis='both', colors='#999999')
        
        # Add statistics box
        stats_text = (
            f"Initial value: {order_parameter[0]:.4f}\n"
            f"Final value: {order_parameter[-1]:.4f}\n"
            f"Mean value: {mean_r:.4f}\n"
            f"Min value: {np.min(order_parameter):.4f}\n"
            f"Max value: {np.max(order_parameter):.4f}"
        )
        
        # Add text with statistics
        ax_order.text(0.02, 0.02, stats_text, transform=ax_order.transAxes,
                  bbox=dict(boxstyle="round,pad=0.6", alpha=0.7, facecolor='#1a1a1a', edgecolor='#444444'),
                  color='white', fontsize=10, verticalalignment='bottom')
        
        # Add legend
        ax_order.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a1a', 
                    edgecolor='#444444', labelcolor='white')
        
        # Add a colorbar for the line gradient
        cbar = fig_order.colorbar(line, ax=ax_order, pad=0.01, location='right')
        cbar.set_label('Time', color='white')
        cbar.ax.yaxis.set_tick_params(color='#999999')
        plt.setp(plt.getp(cbar.ax, 'yticklabels'), color='#999999')
        
        # Display the figure
        st.pyplot(fig_order)
        
        # Create two columns for additional analyses
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            # Order Parameter vs. Oscillator Frequency Relationship
            st.markdown("<h4 class='gradient_text3'>Order Parameter vs. Frequencies</h4>", unsafe_allow_html=True)
            
            fig_freq_r, ax_freq_r = plt.subplots(figsize=(6, 4))
            
            # Set background color
            ax_freq_r.set_facecolor('#1a1a1a')
            fig_freq_r.patch.set_facecolor('#121212')
            
            # Plot natural frequencies vs. final phase coherence
            final_phases = phases[:, -1]
            mean_phase = np.angle(np.mean(np.exp(1j * final_phases)))
            
            # Calculate phase coherence with respect to mean phase
            phase_coherence = np.abs(np.cos(final_phases - mean_phase))
            
            # Create a scatter plot of frequency vs. phase coherence
            scatter = ax_freq_r.scatter(frequencies, phase_coherence, 
                                    c=node_colors, s=100, alpha=0.8, 
                                    edgecolors='white', linewidths=1)
            
            # Add trend line
            try:
                # Try to fit a polynomial
                z = np.polyfit(frequencies, phase_coherence, 2)
                p = np.poly1d(z)
                
                # Generate points for the trend line
                x_trend = np.linspace(min(frequencies), max(frequencies), 100)
                y_trend = p(x_trend)
                
                # Plot the trend line
                ax_freq_r.plot(x_trend, y_trend, color='#ff9966', linewidth=2, 
                           linestyle='--', alpha=0.7, label='Trend')
            except:
                # Skip trend line if fitting fails
                pass
            
            # Add grid
            ax_freq_r.grid(True, color='#333333', alpha=0.3, linestyle=':')
            
            # Set labels and title
            ax_freq_r.set_xlabel('Natural Frequency', color='white', fontsize=12)
            ax_freq_r.set_ylabel('Phase Coherence', color='white', fontsize=12)
            ax_freq_r.set_title('Phase Coherence vs. Natural Frequency', 
                            color='white', fontsize=14, pad=15)
            
            # Make tick labels lighter
            ax_freq_r.tick_params(axis='both', colors='#999999')
            
            # Add legend
            ax_freq_r.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a1a', 
                         edgecolor='#444444', labelcolor='white')
            
            # Display the figure
            st.pyplot(fig_freq_r)
            
            # Add explanation
            st.markdown("""
            <div class='section' style='font-size: 0.9em;'>
                <p>This plot shows how each oscillator's natural frequency relates to its final phase coherence:</p>
                <ul>
                    <li>Oscillators with frequencies closer to the average tend to synchronize better</li>
                    <li>Oscillators with extreme frequencies may remain partially unsynchronized</li>
                </ul>
                <p>Phase coherence is measured as cos(θ<sub>i</sub> - ψ), where ψ is the mean phase.</p>
            </div>
            """, unsafe_allow_html=True)
            
        with analysis_col2:
            # Phase Distribution at final time
            st.markdown("<h4 class='gradient_text3'>Final Phase Distribution</h4>", unsafe_allow_html=True)
            
            # Create polar histogram of final phases
            fig_polar_hist = plt.figure(figsize=(6, 4))
            ax_polar = fig_polar_hist.add_subplot(111, projection='polar')
            
            # Get final phases
            final_phases = phases[:, -1] % (2 * np.pi)
            
            # Set number of bins based on oscillator count
            n_bins = min(36, max(12, sim_n_oscillators // 2))
            
            # Create histogram
            hist, bin_edges = np.histogram(final_phases, bins=n_bins, range=(0, 2*np.pi))
            
            # Get bin centers
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Create colormap
            colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(hist)))
            
            # Plot bars
            bars = ax_polar.bar(bin_centers, hist, width=2*np.pi/n_bins, 
                           bottom=0.0, alpha=0.7, color=colors)
            
            # Add mean phase marker
            mean_phase = np.angle(np.mean(np.exp(1j * final_phases)))
            ax_polar.plot([mean_phase, mean_phase], [0, max(hist)*1.2], 
                      color='red', linewidth=2, alpha=0.7, label='Mean Phase')
            
            # Set background color
            ax_polar.set_facecolor('#1a1a1a')
            fig_polar_hist.patch.set_facecolor('#121212')
            
            # Make tick labels lighter
            ax_polar.tick_params(axis='both', colors='#999999')
            
            # Set title
            ax_polar.set_title('Final Phase Distribution', color='white', fontsize=14, pad=15)
            
            # Add legend
            ax_polar.legend(loc='upper right', framealpha=0.8, facecolor='#1a1a1a', 
                        edgecolor='#444444', labelcolor='white')
            
            # Display the figure
            st.pyplot(fig_polar_hist)
            
            # Add explanation
            st.markdown("""
            <div class='section' style='font-size: 0.9em;'>
                <p>This polar histogram shows the final distribution of oscillator phases:</p>
                <ul>
                    <li>In fully synchronized systems, all oscillators cluster at a single phase</li>
                    <li>In partially synchronized systems, oscillators form a non-uniform distribution</li>
                    <li>The red line indicates the mean phase of all oscillators</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Create frequency distribution box
            st.markdown("<h4 class='gradient_text3'>Frequency Statistics</h4>", unsafe_allow_html=True)
            
            # Calculate statistics
            freq_mean = np.mean(frequencies)
            freq_std = np.std(frequencies)
            freq_min = np.min(frequencies)
            freq_max = np.max(frequencies)
            freq_range = freq_max - freq_min
            
            # Create a table of frequency statistics
            st.markdown(f"""
            <div class='section'>
                <table style="width:100%; border-collapse: collapse; font-size: 0.9em;">
                    <tr>
                        <th style="text-align: left; padding: 8px; border-bottom: 1px solid #444;">Statistic</th>
                        <th style="text-align: right; padding: 8px; border-bottom: 1px solid #444;">Value</th>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Mean Frequency</td>
                        <td style="text-align: right; padding: 8px;">{freq_mean:.4f}</td>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Standard Deviation</td>
                        <td style="text-align: right; padding: 8px;">{freq_std:.4f}</td>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Minimum Frequency</td>
                        <td style="text-align: right; padding: 8px;">{freq_min:.4f}</td>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Maximum Frequency</td>
                        <td style="text-align: right; padding: 8px;">{freq_max:.4f}</td>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Frequency Range</td>
                        <td style="text-align: right; padding: 8px;">{freq_range:.4f}</td>
                    </tr>
                    <tr>
                        <td style="text-align: left; padding: 8px;">Critical Coupling (est.)</td>
                        <td style="text-align: right; padding: 8px;">{freq_std * 2:.4f}</td>
                    </tr>
                </table>
            </div>
            """, unsafe_allow_html=True)
            
            # If coupling strength is below the critical value, add a note
            critical_k = freq_std * 2  # Rough estimate for all-to-all coupling
            if coupling_strength < critical_k:
                st.markdown(f"""
                <div style="background-color: rgba(255, 100, 100, 0.2); padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <p><b>Note:</b> The coupling strength ({coupling_strength:.2f}) is below the estimated critical value ({critical_k:.2f}) 
                    for full synchronization in an all-to-all network. This may result in partial or no synchronization.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background-color: rgba(100, 255, 100, 0.2); padding: 10px; border-radius: 5px; margin-top: 10px;">
                    <p><b>Note:</b> The coupling strength ({coupling_strength:.2f}) is above the estimated critical value ({critical_k:.2f}) 
                    for an all-to-all network, which favors synchronization. Network structure may modify this threshold.</p>
                </div>
                """, unsafe_allow_html=True)

# Run the application
if __name__ == "__main__":
    main()