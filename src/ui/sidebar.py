"""
Sidebar UI components for Kuramoto Model Simulator.
"""
import streamlit as st
import numpy as np
from src.models.kuramoto_model import KuramotoModel

def oscillator_parameters():
    """Render oscillator and coupling parameters section."""
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
    
    return n_oscillators, coupling_strength

def frequency_distribution_parameters():
    """Render frequency distribution parameters section."""
    # Frequency distribution type
    freq_type = st.sidebar.selectbox(
        "Frequency Distribution",
        ["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"],
        index=["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"].index(st.session_state.freq_type) 
              if st.session_state.freq_type in ["Normal", "Uniform", "Bimodal", "Golden Ratio", "Custom"] else 0,
        help="Distribution of natural frequencies",
        key="freq_type"
    )
    
    # Get the number of oscillators
    n_oscillators = st.session_state.n_oscillators
    
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
    
    return freq_type, frequencies

def simulation_time_parameters():
    """Render simulation time parameters section."""
    # Simulation time parameters
    simulation_time = st.sidebar.slider(
        "Simulation Time",
        min_value=1.0,
        max_value=100.0,
        step=1.0,
        help="Total simulation time",
        key="simulation_time"
    )
    
    # Time Step Controls
    time_step = st.sidebar.slider(
        "Time Step",
        min_value=0.001,
        max_value=0.1,
        step=0.001,
        format="%.3f",
        help="Time step for simulation (smaller = more accurate but slower)",
        key="time_step"
    )
    
    # Time step optimization controls - smaller buttons with adjusted size
    time_step_col1, time_step_col2 = st.sidebar.columns([1, 1])
    
    # Add an auto-optimize time step button in column 1 with smaller text
    if time_step_col1.button("🧠 Optimize", help="Automatically calculate optimal time step for stability and accuracy"):
        # Need to get the random seed value from session state before using it
        current_random_seed = st.session_state.random_seed if "random_seed" in st.session_state else 42
        
        # Check for adjacency matrix in session state
        adjacency_matrix_for_optimization = None
        if 'loaded_adj_matrix' in st.session_state:
            adjacency_matrix_for_optimization = st.session_state.loaded_adj_matrix
            print(f"Using adjacency matrix from session state for optimization, shape: {adjacency_matrix_for_optimization.shape if hasattr(adjacency_matrix_for_optimization, 'shape') else 'unknown'}")
        
        # Get frequencies from session state based on frequency type
        if st.session_state.freq_type == "Normal":
            frequencies = np.random.normal(
                st.session_state.freq_mean, 
                st.session_state.freq_std, 
                st.session_state.n_oscillators
            )
        elif st.session_state.freq_type == "Uniform":
            frequencies = np.random.uniform(
                st.session_state.freq_min, 
                st.session_state.freq_max, 
                st.session_state.n_oscillators
            )
        elif st.session_state.freq_type == "Bimodal":
            mix = np.random.choice([0, 1], size=st.session_state.n_oscillators)
            freq1 = np.random.normal(st.session_state.peak1, 0.3, st.session_state.n_oscillators)
            freq2 = np.random.normal(st.session_state.peak2, 0.3, st.session_state.n_oscillators)
            frequencies = mix * freq1 + (1 - mix) * freq2
        elif st.session_state.freq_type == "Golden Ratio":
            phi = (1 + 5**0.5) / 2
            golden_ratio_start = -3.0
            frequencies = np.array([golden_ratio_start + i * phi for i in range(st.session_state.n_oscillators)])
        else:  # Custom
            try:
                frequencies = np.array([float(x.strip()) for x in st.session_state.custom_freqs.split(',')])
                if len(frequencies) < st.session_state.n_oscillators:
                    frequencies = np.tile(frequencies, int(np.ceil(st.session_state.n_oscillators / len(frequencies))))
                frequencies = frequencies[:st.session_state.n_oscillators]
            except:
                frequencies = np.random.normal(0, 1, st.session_state.n_oscillators)
                
        # Create a temporary model to calculate optimal time step
        temp_model = KuramotoModel(
            n_oscillators=st.session_state.n_oscillators,
            coupling_strength=st.session_state.coupling_strength,
            frequencies=frequencies,
            simulation_time=st.session_state.simulation_time,
            time_step=st.session_state.time_step,
            random_seed=current_random_seed,
            adjacency_matrix=adjacency_matrix_for_optimization
        )
        
        # Get the optimization results
        optimization_results = temp_model.compute_optimal_time_step(safety_factor=0.85)
        
        # Store the optimized time step in a different session state variable
        # Rather than directly changing the slider's value
        st.session_state.optimized_time_step = optimization_results['optimal_time_step']
        
        # Display a success message with the explanation
        st.sidebar.success(f"""
        Time step optimized to {optimization_results['optimal_time_step']:.4f}
        
        Please manually set this value in the Time Step slider above.
        """)
        
        # Display detailed optimization information in an expander
        with st.sidebar.expander("Optimization Details"):
            st.markdown(f"""
            **Stability Level:** {optimization_results['stability_level']}  
            **Accuracy Level:** {optimization_results['accuracy_level']}  
            **Computation Efficiency:** {optimization_results['computation_level']}
            
            {optimization_results['explanation']}
            """)
        
        # We don't rerun because we can't automatically update the slider widget
    
    # Add checkbox in column 2 to always auto-optimize on simulation run
    auto_optimize_on_run = time_step_col2.checkbox(
        "Auto on Run", 
        value=st.session_state.auto_optimize_on_run,
        help="Automatically optimize time step each time the simulation runs",
        key="auto_optimize_on_run"
    )
    
    return simulation_time, time_step

def random_seed_parameter():
    """Render random seed parameter section."""
    random_seed = int(st.sidebar.number_input(
        "Random Seed", 
        min_value=0,
        step=1,
        help="Seed for reproducibility",
        key="random_seed"
    ))
    
    return random_seed

def network_connectivity_parameters():
    """Render network connectivity parameters section."""
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
            # If the number of oscillators is different than the matrix dimensions, we should adjust
            if adj_matrix.shape[0] != st.session_state.n_oscillators:
                # Queue the update for oscillator count - can't update it directly due to Streamlit limitations
                st.session_state.next_n_oscillators = adj_matrix.shape[0]
                st.sidebar.warning(f"""
                Matrix dimensions ({adj_matrix.shape[0]}×{adj_matrix.shape[1]}) don't match oscillator count. 
                Oscillator count will be adjusted on next run.
                """)
    
    # Custom adjacency matrix input
    if network_type == "Custom Adjacency Matrix":
        st.sidebar.markdown("<h4>Custom Adjacency Matrix</h4>", unsafe_allow_html=True)
        
        # Add help text
        st.sidebar.markdown("""
        <div style="background-color: rgba(100,100,100,0.2); padding: 8px; border-radius: 5px; margin-bottom: 10px;">
        <p style="margin-bottom: 5px;"><b>Format instructions:</b></p>
        <p style="margin-bottom: 5px; font-size: 0.9em;">
        • Enter values separated by commas (rows) and line breaks (columns)<br>
        • Matrix must be square (same number of rows and columns)<br>
        • 1 means connected, 0 means not connected<br>
        • Diagonal should be 0 (no self-loops)
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text area for matrix input with monospace font
        adj_matrix_input = st.sidebar.text_area(
            "Enter adjacency matrix",
            value=st.session_state.adj_matrix_input,
            height=150,
            key="adj_matrix_input",
            help="Input format: comma-separated values, line breaks for rows"
        )
        
        try:
            # Parse the input string to a 2D array
            row_strings = adj_matrix_input.strip().split('\n')
            rows = []
            for row_str in row_strings:
                row = [int(float(x.strip())) for x in row_str.split(',')]
                rows.append(row)
            
            # Convert to numpy array
            parsed_matrix = np.array(rows)
            
            # Verify it's a valid adjacency matrix
            if parsed_matrix.shape[0] != parsed_matrix.shape[1]:
                st.sidebar.error(f"Matrix must be square. Current shape: {parsed_matrix.shape}")
            else:
                # Ensure diagonal is zero (no self-loops)
                np.fill_diagonal(parsed_matrix, 0)
                
                # Store the parsed matrix in session state for use in simulation
                st.session_state.loaded_adj_matrix = parsed_matrix
                
                # Show matrix dimensions
                st.sidebar.markdown(f"""
                <div style="background-color: rgba(0,150,0,0.1); padding: 5px; border-radius: 5px;">
                <p>Matrix dimensions: {parsed_matrix.shape[0]}×{parsed_matrix.shape[1]}</p>
                <p>Number of connections: {np.count_nonzero(parsed_matrix)}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # If the number of oscillators doesn't match matrix dimensions, we need to adjust
                if parsed_matrix.shape[0] != st.session_state.n_oscillators:
                    # Queue the update for oscillator count - can't update it directly due to Streamlit limitations
                    st.session_state.next_n_oscillators = parsed_matrix.shape[0]
                    st.sidebar.warning(f"""
                    Matrix dimensions ({parsed_matrix.shape[0]}×{parsed_matrix.shape[1]}) don't match oscillator count ({st.session_state.n_oscillators}). 
                    Oscillator count will be adjusted on next run.
                    """)
                
                adj_matrix = parsed_matrix
                
        except Exception as e:
            st.sidebar.error(f"Error parsing matrix: {str(e)}")
            # Reset the session state if we can't parse the matrix
            if 'loaded_adj_matrix' in st.session_state:
                del st.session_state.loaded_adj_matrix
    else:
        # For non-custom matrix types, clear the loaded matrix from session state
        if 'loaded_adj_matrix' in st.session_state:
            del st.session_state.loaded_adj_matrix
    
    # Random network parameters
    if network_type == "Random":
        connection_probability = st.sidebar.slider(
            "Connection Probability",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Probability of connection between oscillators",
            key="connection_probability"
        )
    else:
        connection_probability = 0.5  # Default value
    
    return network_type, adj_matrix, connection_probability