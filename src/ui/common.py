"""
Common UI utilities and styling for the Kuramoto Model Simulator.
"""
import streamlit as st
import matplotlib.pyplot as plt
import base64
import json
import numpy as np

def setup_page():
    """Configure the Streamlit page settings and apply custom styling."""
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
    with open("static/css/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # Get the base64 encoded image
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

def display_title():
    """Display the application title."""
    st.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)

def display_sidebar_header():
    """Display the sidebar header."""
    st.sidebar.markdown("<h2 class='gradient_text1'>Simulation Parameters</h2>", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    # Initialize session state for configuration loading
    if 'loaded_config' not in st.session_state:
        st.session_state.loaded_config = None
    
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
    
    # For auto-optimizing time step
    if "auto_optimize_on_run" not in st.session_state:
        st.session_state.auto_optimize_on_run = False
        
    # Initialize time_idx for animation to avoid starting with 399
    if "time_idx" not in st.session_state:
        st.session_state.time_idx = 0

def load_configuration():
    """Load configuration if available in session state."""
    # Apply loaded configuration if available
    if st.session_state.loaded_config is not None:
        config = st.session_state.loaded_config
        
        # Update session state with configuration values
        st.session_state.n_oscillators = config['n_oscillators']
        st.session_state.coupling_strength = config['coupling_strength']
        st.session_state.simulation_time = config['simulation_time']
        st.session_state.time_step = config['time_step']
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

def update_oscillator_count():
    """Update oscillator count if pending update exists."""
    if st.session_state.next_n_oscillators is not None:
        print(f"Updating oscillator count from {st.session_state.n_oscillators} to {st.session_state.next_n_oscillators}")
        st.session_state.n_oscillators = st.session_state.next_n_oscillators
        st.session_state.next_n_oscillators = None  # Clear the pending update