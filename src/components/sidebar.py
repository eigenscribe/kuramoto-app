"""
Sidebar components for the Kuramoto Model Simulator.
"""

import streamlit as st
import numpy as np
import json
from src.database.database import save_configuration
from src.utils.json_handler import parse_json_parameters, generate_example_json, generate_small_world_example

def render_sidebar(frequencies_callback):
    """
    Render the sidebar with all configuration controls.
    
    Args:
        frequencies_callback: Function that returns frequencies based on UI settings
        
    Returns:
        tuple: (n_oscillators, coupling_strength, frequencies, network_type, 
                simulation_time, random_seed, adj_matrix)
    """
    # Add main title with gradient text and image
    st.sidebar.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)
    
    # Create subheading for manual parameters
    st.sidebar.markdown("<h3 class='gradient_text1'>Manual Configuration</h3>", unsafe_allow_html=True)
    
    # For auto-adjusting oscillator count based on matrix dimensions
    _process_oscillator_count_update()
    
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
    
    # Get frequencies from callback
    frequencies = frequencies_callback(freq_type, n_oscillators)
    
    # Random seed setting
    if "random_seed" not in st.session_state:
        st.session_state.random_seed = 42
    
    random_seed = int(st.sidebar.number_input(
        "Random Seed", 
        min_value=0,
        step=1,
        help="Seed for reproducibility",
        key="random_seed"
    ))
    
    # Network Connectivity Configuration
    st.sidebar.markdown("<h3 class='gradient_text1'>Network Connectivity</h3>", unsafe_allow_html=True)
    
    # Add a refresh button at the top of the Network Connectivity section
    if st.sidebar.button("🔄 Refresh Simulation", key="refresh_btn"):
        st.session_state.refresh_network = True
        print("Network refresh requested via main refresh button")
        st.rerun()
    
    # Network type selection
    network_type, adj_matrix = _render_network_type_selection(n_oscillators, coupling_strength, 
                                                            random_seed, freq_type, simulation_time=st.session_state.simulation_time)
    
    # Add JSON Configuration section at the bottom of the sidebar
    _render_json_section()
    
    # Time Controls
    simulation_time = st.sidebar.slider(
        "Simulation Time",
        min_value=10.0,
        max_value=200.0,
        step=10.0,
        help="Duration of the simulation",
        key="simulation_time"
    )
    
    return n_oscillators, coupling_strength, frequencies, network_type, simulation_time, random_seed, adj_matrix

def _process_oscillator_count_update():
    """Process any pending oscillator count updates."""
    # If we have a pending oscillator count update from the previous run, apply it now
    if st.session_state.next_n_oscillators is not None:
        print(f"Updating oscillator count from {st.session_state.n_oscillators} to {st.session_state.next_n_oscillators}")
        st.session_state.n_oscillators = st.session_state.next_n_oscillators
        st.session_state.next_n_oscillators = None  # Clear the pending update

def _render_network_type_selection(n_oscillators, coupling_strength, random_seed, freq_type, simulation_time):
    """Render the network type selection portion of the sidebar."""
    # Create radio button for network type without specifying both index and session state key
    options = ["All-to-All", "Nearest Neighbor", "Random", "Custom Adjacency Matrix"]
    if "network_type" not in st.session_state:
        # Only set this if it's not already in session state
        st.session_state.network_type = "Random"
    
    # Get current index
    current_index = options.index(st.session_state.network_type)
    
    # Create the radio button
    network_type = st.sidebar.radio(
        "Network Type",
        options=options,
        index=current_index,
        help="Define how oscillators are connected to each other"
    )
    
    # Update session state if changed
    if network_type != st.session_state.network_type:
        st.session_state.network_type = network_type
    
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
        adj_matrix = _render_custom_adjacency_matrix(n_oscillators, coupling_strength, 
                                                    random_seed, freq_type, simulation_time)
    
    return network_type, adj_matrix

def _render_custom_adjacency_matrix(n_oscillators, coupling_strength, random_seed, freq_type, simulation_time):
    """Render the custom adjacency matrix input section."""
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
    if not st.session_state.adj_matrix_input:
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
    adj_matrix = None
    if adj_matrix_input:
        try:
            # Parse the input text into a numpy array
            rows = adj_matrix_input.strip().split('\n')
            
            # Ensure we have at least one row
            if len(rows) == 0:
                raise ValueError("No data found in matrix input")
                
            # Process each row, removing extra spaces and parsing values
            parsed_matrix = []
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
                    parsed_matrix.append(values)
            
            # Make sure we have a valid matrix with data
            if not parsed_matrix:
                raise ValueError("Could not find valid numeric data in input")
                
            # Convert to numpy array for faster processing
            adj_matrix = np.array(parsed_matrix)
            
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
                            # Get frequency parameters based on type
                            freq_params = {}
                            if freq_type == "Normal":
                                freq_params = {
                                    "mean": float(st.session_state.freq_mean),
                                    "std": float(st.session_state.freq_std)
                                }
                            elif freq_type == "Uniform":
                                freq_params = {
                                    "min": float(st.session_state.freq_min),
                                    "max": float(st.session_state.freq_max)
                                }
                                
                            # Save the configuration with current parameters
                            config_id = save_configuration(
                                name=preset_name,
                                n_oscillators=adj_matrix.shape[0],
                                coupling_strength=coupling_strength,
                                simulation_time=simulation_time,
                                time_step=0.01,  # Default value for backward compatibility
                                random_seed=random_seed,
                                network_type="Custom Adjacency Matrix",
                                frequency_distribution=freq_type,
                                frequency_params=json.dumps(freq_params),
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
    
    return adj_matrix

def _render_json_section():
    """Render the JSON configuration section of the sidebar."""
    st.sidebar.markdown("<hr style='margin: 15px 0px; border-color: rgba(255,255,255,0.2);'>", unsafe_allow_html=True)
    st.sidebar.markdown("<h3 class='gradient_text1'>JSON Configuration</h3>", unsafe_allow_html=True)
    
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
        example_json = generate_example_json()
        st.code(json.dumps(example_json, indent=2), language="json")
        
        # Add small-world network example
        st.markdown("**Small-world network example:**")
        complex_example = generate_small_world_example()
        
        # Add button to use this example
        if st.button("Use Small-World", key="small_world_btn"):
            st.session_state.json_example = json.dumps(complex_example, indent=2)
            st.rerun()
    
    # Add import and export buttons
    col1, col2 = st.sidebar.columns(2)
    
    # Import button logic
    with col1:
        if st.button("Import", key="import_btn"):
            if json_input:
                params, error = parse_json_parameters(json_input)
                if error:
                    st.error(f"Error parsing JSON: {error}")
                else:
                    # Store in session state for next run to apply
                    st.session_state.temp_imported_params = params
                    print("Imported parameters successfully, stored in session state")
                    st.success("Parameters imported successfully!")
                    st.rerun()  # Force rerun to apply the imported parameters
    
    # Export button logic - create and display current parameters as JSON
    with col2:
        if st.button("Export", key="export_btn"):
            # Create a dictionary with current parameters
            current_params = {
                "n_oscillators": st.session_state.n_oscillators,
                "coupling_strength": st.session_state.coupling_strength,
                "network_type": st.session_state.network_type,
                "simulation_time": st.session_state.simulation_time,
                "random_seed": st.session_state.random_seed,
                "frequency_distribution": st.session_state.freq_type
            }
            
            # Add frequency parameters based on distribution type
            if st.session_state.freq_type == "Normal":
                current_params["frequency_parameters"] = {
                    "mean": st.session_state.freq_mean,
                    "std": st.session_state.freq_std
                }
            elif st.session_state.freq_type == "Uniform":
                current_params["frequency_parameters"] = {
                    "min": st.session_state.freq_min,
                    "max": st.session_state.freq_max
                }
            elif st.session_state.freq_type == "Bimodal":
                current_params["frequency_parameters"] = {
                    "peak1": st.session_state.peak1,
                    "peak2": st.session_state.peak2
                }
            elif st.session_state.freq_type == "Custom":
                try:
                    custom_values = [float(x.strip()) for x in st.session_state.custom_freqs.split(',')]
                    current_params["frequency_parameters"] = {
                        "custom_values": custom_values
                    }
                except:
                    current_params["frequency_parameters"] = {}
            
            # Add adjacency matrix if using custom network type
            if st.session_state.network_type == "Custom Adjacency Matrix" and 'loaded_adj_matrix' in st.session_state:
                try:
                    matrix = st.session_state.loaded_adj_matrix
                    if matrix is not None and hasattr(matrix, 'tolist'):
                        current_params["adjacency_matrix"] = matrix.tolist()
                except Exception as e:
                    print(f"Could not include adjacency matrix in export: {str(e)}")
            
            # Convert to JSON string and update text area
            st.session_state.json_example = json.dumps(current_params, indent=2)
            print("Exported current parameters to JSON")
            st.rerun()  # Force rerun to update the text area