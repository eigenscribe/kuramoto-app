"""
Machine Learning tab UI components for Kuramoto Model Simulator.
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from scipy.interpolate import interp1d
import matplotlib.cm as cm

from src.utils.ml_helper import (
    create_critical_coupling_dataset,
    create_network_topology_dataset,
    visualize_dataset,
    export_for_pytorch,
    list_ml_datasets
)
from src.utils.database import (
    get_ml_dataset,
    export_ml_dataset
)

def render_ml_tab():
    """Render the Machine Learning tab UI."""
    st.markdown("<h2 class='gradient_text1'>Machine Learning</h2>", unsafe_allow_html=True)
    
    # Add an info box explaining the machine learning capabilities
    st.markdown("""
    <div style="background-color: rgba(0,0,0,0.2); padding: 15px; border-radius: 5px; margin-bottom: 20px;">
    <h4 style="margin-top: 0;">Machine Learning with the Kuramoto Model</h4>
    <p>This tab allows you to create and manage datasets for machine learning, as well as visualize the results
    of machine learning models trained on Kuramoto simulation data.</p>
    <p>You can create datasets for:</p>
    <ul>
        <li>Predicting critical coupling strength from oscillator parameters</li>
        <li>Comparing synchronization dynamics across different network topologies</li>
        <li>Analyzing the relationship between natural frequencies and synchronization</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Import ML helper functions
    
    # Create tabs for different ML functionalities
    ml_tab1, ml_tab2, ml_tab3 = st.tabs(["Create Datasets", "Manage Datasets", "Visualize Results"])
    
    # TAB 1: CREATE DATASETS
    with ml_tab1:
        st.markdown("<h3 class='gradient_text1'>Create Machine Learning Datasets</h3>", unsafe_allow_html=True)
        
        # Dataset type selection
        dataset_type = st.selectbox(
            "Dataset Type",
            ["Critical Coupling Analysis", "Network Topology Comparison"]
        )
        
        if dataset_type == "Critical Coupling Analysis":
            st.markdown("""
            <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                <p>Generate a dataset to study the transition to synchronization at critical coupling strength.
                This creates multiple simulations around the theoretical critical coupling point.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Parameters for critical coupling dataset
            cc_col1, cc_col2 = st.columns(2)
            
            with cc_col1:
                cc_dataset_name = st.text_input(
                    "Dataset Name", 
                    value="critical_coupling_dataset", 
                    key="cc_dataset_name"
                )
                cc_n_oscillators = st.slider(
                    "Number of Oscillators",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key="cc_n_oscillators"
                )
            
            with cc_col2:
                cc_freq_std = st.slider(
                    "Frequency Standard Deviation",
                    min_value=0.1,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    key="cc_freq_std"
                )
                cc_k_points = st.slider(
                    "Number of K Values",
                    min_value=10,
                    max_value=50,
                    value=20,
                    step=5,
                    key="cc_k_points"
                )
            
            # Calculate theoretical critical coupling
            critical_k = 2 * cc_freq_std / np.pi
            st.markdown(f"""
            <div style='padding: 10px; background-color: rgba(0,100,255,0.1); border-radius: 5px;'>
                <p>Theoretical critical coupling value: <b>{critical_k:.4f}</b></p>
                <p>The dataset will sample points around this value to capture the transition to synchronization.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # K values range
            k_min = st.number_input(
                "Minimum K Value",
                value=float(0.2 * critical_k),
                format="%.2f",
                key="cc_k_min"
            )
            k_max = st.number_input(
                "Maximum K Value",
                value=float(3.0 * critical_k),
                format="%.2f",
                key="cc_k_max"
            )
            
            # Calculate K values
            if 'cc_k_values' not in st.session_state or st.session_state.cc_k_values is None:
                st.session_state.cc_k_values = np.linspace(k_min, k_max, cc_k_points)
            
            # Button to generate dataset
            if st.button("Generate Critical Coupling Dataset", type="primary", key="gen_cc_dataset"):
                try:
                    with st.spinner("Generating dataset... This may take a while."):
                        # Create the dataset
                        dataset_id = create_critical_coupling_dataset(
                            name=cc_dataset_name,
                            n_oscillators=cc_n_oscillators,
                            frequency_std=cc_freq_std,
                            k_values=st.session_state.cc_k_values
                        )
                        
                        st.success(f"Dataset '{cc_dataset_name}' created successfully with ID: {dataset_id}")
                        st.info("You can now visualize this dataset in the 'Manage Datasets' tab.")
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")
        
        elif dataset_type == "Network Topology Comparison":
            st.markdown("""
            <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
                <p>Generate a dataset to compare different network topologies (all-to-all, ring, random) 
                with identical oscillator parameters.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Parameters for network topology dataset
            nt_col1, nt_col2 = st.columns(2)
            
            with nt_col1:
                nt_dataset_name = st.text_input(
                    "Dataset Name", 
                    value="network_topology_dataset", 
                    key="nt_dataset_name"
                )
                nt_n_oscillators = st.slider(
                    "Number of Oscillators",
                    min_value=5,
                    max_value=50,
                    value=10,
                    step=5,
                    key="nt_n_oscillators"
                )
            
            with nt_col2:
                nt_coupling = st.slider(
                    "Coupling Strength",
                    min_value=0.1,
                    max_value=5.0,
                    value=1.0,
                    step=0.1,
                    key="nt_coupling"
                )
                nt_samples = st.slider(
                    "Samples Per Network Type",
                    min_value=5,
                    max_value=30,
                    value=10,
                    step=5,
                    key="nt_samples"
                )
            
            st.markdown(f"""
            <div style='padding: 10px; background-color: rgba(0,180,0,0.1); border-radius: 5px;'>
                <p>This will generate <b>{nt_samples * 3}</b> total simulations:</p>
                <ul>
                    <li>{nt_samples} with all-to-all connectivity</li>
                    <li>{nt_samples} with ring connectivity</li>
                    <li>{nt_samples} with random connectivity</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
            # Button to generate dataset
            if st.button("Generate Network Topology Dataset", type="primary", key="gen_nt_dataset"):
                try:
                    with st.spinner("Generating dataset... This may take a while."):
                        # Create the dataset
                        dataset_id = create_network_topology_dataset(
                            name=nt_dataset_name,
                            n_oscillators=nt_n_oscillators,
                            coupling_strength=nt_coupling,
                            n_samples=nt_samples
                        )
                        
                        st.success(f"Dataset '{nt_dataset_name}' created successfully with ID: {dataset_id}")
                        st.info("You can now visualize this dataset in the 'Manage Datasets' tab.")
                except Exception as e:
                    st.error(f"Error creating dataset: {str(e)}")
    
    # TAB 2: MANAGE DATASETS
    with ml_tab2:
        st.markdown("<h3 class='gradient_text1'>Manage ML Datasets</h3>", unsafe_allow_html=True)
        
        # Refresh datasets list
        if st.button("🔄 Refresh Datasets", key="refresh_datasets"):
            st.rerun()
        
        # Get all available datasets
        datasets = list_ml_datasets()
        
        if not datasets:
            st.info("No machine learning datasets found. Create a dataset in the 'Create Datasets' tab.")
        else:
            # Display datasets in a table
            datasets_df = pd.DataFrame([
                {
                    "ID": dataset["id"],
                    "Name": dataset["name"],
                    "Description": dataset["description"] if dataset["description"] else "",
                    "Type": dataset["feature_type"],
                    "Created": dataset["timestamp"].strftime("%Y-%m-%d %H:%M"),
                    "Simulations": len(dataset["simulations"])
                } for dataset in datasets
            ])
            
            st.dataframe(datasets_df, use_container_width=True)
            
            # Dataset action section
            st.markdown("<h4 class='gradient_text1'>Dataset Actions</h4>", unsafe_allow_html=True)
            
            # Select a dataset to work with
            dataset_options = [f"ID: {dataset['id']} - {dataset['name']}" for dataset in datasets]
            selected_dataset = st.selectbox("Select Dataset", dataset_options)
            
            if selected_dataset:
                # Extract dataset ID from selection
                dataset_id = int(selected_dataset.split("ID: ")[1].split(" -")[0])
                
                # Create columns for different actions
                action_col1, action_col2 = st.columns(2)
                
                with action_col1:
                    st.markdown("<h5>Visualize</h5>", unsafe_allow_html=True)
                    if st.button("View Dataset Visualizations", key="view_dataset", use_container_width=True):
                        try:
                            with st.spinner("Generating visualizations..."):
                                # Use the visualization function from ml_helper
                                st.markdown("<h4 class='gradient_text2'>Dataset Visualizations</h4>", unsafe_allow_html=True)
                                visualize_dataset(dataset_id)
                        except Exception as e:
                            st.error(f"Error visualizing dataset: {str(e)}")
                
                with action_col2:
                    st.markdown("<h5>Export</h5>", unsafe_allow_html=True)
                    export_format = st.selectbox(
                        "Export Format",
                        ["PyTorch", "NumPy", "Pandas CSV"],
                        key="export_format"
                    )
                    
                    if st.button("Export Dataset", key="export_dataset", use_container_width=True):
                        try:
                            with st.spinner("Exporting dataset..."):
                                if export_format == "PyTorch":
                                    export_path = export_for_pytorch(dataset_id)
                                    st.success(f"Dataset exported for PyTorch: {export_path}")
                                else:
                                    # For other formats, use the general export function
                                    format_map = {
                                        "NumPy": "numpy",
                                        "Pandas CSV": "pandas"
                                    }
                                    export_path = export_ml_dataset(
                                        dataset_id=dataset_id,
                                        format=format_map[export_format]
                                    )
                                    st.success(f"Dataset exported in {export_format} format: {export_path}")
                        except Exception as e:
                            st.error(f"Error exporting dataset: {str(e)}")
                
                # Additional dataset information
                st.markdown("<h4 class='gradient_text1'>Dataset Details</h4>", unsafe_allow_html=True)
                
                try:
                    # Get detailed information about the selected dataset
                    dataset_details = get_ml_dataset(dataset_id)
                    
                    if dataset_details:
                        # Display dataset metadata
                        st.markdown(f"""
                        <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                            <p><b>Name:</b> {dataset_details['name']}</p>
                            <p><b>Description:</b> {dataset_details['description'] or 'No description'}</p>
                            <p><b>Feature Type:</b> {dataset_details['feature_type']}</p>
                            <p><b>Target Type:</b> {dataset_details['target_type']}</p>
                            <p><b>Created:</b> {dataset_details['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</p>
                            <p><b>Simulations:</b> {len(dataset_details['simulations'])}</p>
                            <p><b>Features:</b> {len(dataset_details['features'])}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display features information
                        if dataset_details['features']:
                            st.markdown("<h5>Features</h5>", unsafe_allow_html=True)
                            features_df = pd.DataFrame([
                                {
                                    "Name": feature['name'],
                                    "Type": feature['feature_type'],
                                    "Description": feature['description'] or 'No description'
                                } for feature in dataset_details['features']
                            ])
                            st.dataframe(features_df, use_container_width=True)
                except Exception as e:
                    st.error(f"Error fetching dataset details: {str(e)}")
    
    # TAB 3: VISUALIZE RESULTS
    with ml_tab3:
        st.markdown("<h3 class='gradient_text1'>Visualization Tools</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px; margin-bottom: 15px;'>
            <p>Explore interactive visualizations of machine learning results and datasets.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Section for selecting a dataset to visualize
        if not datasets:
            st.info("No machine learning datasets found. Create a dataset in the 'Create Datasets' tab.")
        else:
            # Allow user to select a dataset
            vis_dataset_options = [f"ID: {dataset['id']} - {dataset['name']}" for dataset in datasets]
            vis_selected_dataset = st.selectbox("Select Dataset to Visualize", vis_dataset_options, key="vis_dataset")
            
            if vis_selected_dataset:
                # Extract dataset ID
                vis_dataset_id = int(vis_selected_dataset.split("ID: ")[1].split(" -")[0])
                
                # Get dataset details
                dataset_details = get_ml_dataset(vis_dataset_id)
                
                if dataset_details:
                    # Visualization type selection
                    vis_type = st.selectbox(
                        "Visualization Type",
                        ["Parameter Sweep", "Critical Coupling Analysis", "Network Type Comparison"],
                        key="vis_type"
                    )
                    
                    if vis_type == "Parameter Sweep":
                        st.markdown("<h4>Parameter Sweep Visualization</h4>", unsafe_allow_html=True)
                        
                        # Visualize how a parameter affects synchronization
                        try:
                            # Extract features needed for this visualization
                            if any(f['name'] == 'coupling_strength' for f in dataset_details['features']) and \
                               any(f['name'] == 'steady_state_sync' for f in dataset_details['features']):
                                
                                with st.spinner("Generating visualization..."):
                                    # Create a scatter plot of K vs r
                                    for feature in dataset_details['features']:
                                        if feature['name'] == 'coupling_strength':
                                            k_values = feature['data']['value']
                                        elif feature['name'] == 'steady_state_sync':
                                            r_values = feature['data']['value']
                                    
                                    # Sort by coupling strength
                                    sort_idx = np.argsort(k_values)
                                    k_sorted = k_values[sort_idx]
                                    r_sorted = r_values[sort_idx]
                                    
                                    # Create the plot
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    ax.scatter(k_sorted, r_sorted, c=k_sorted, cmap='viridis', s=50, alpha=0.8)
                                    ax.plot(k_sorted, r_sorted, 'k--', alpha=0.3)
                                    
                                    # Format the plot
                                    ax.set_xlabel('Coupling Strength (K)', fontsize=12)
                                    ax.set_ylabel('Order Parameter (r)', fontsize=12)
                                    ax.set_title('Synchronization vs. Coupling Strength', fontsize=14)
                                    ax.grid(True, alpha=0.3)
                                    
                                    # Estimate critical coupling
                                    if len(k_sorted) > 5:
                                        # Smooth the curve
                                        k_smooth = np.linspace(k_sorted.min(), k_sorted.max(), 100)
                                        r_smooth = interp1d(k_sorted, r_sorted, kind='cubic')(k_smooth)
                                        
                                        # Find the point of steepest slope
                                        slopes = np.gradient(r_smooth, k_smooth)
                                        critical_idx = np.argmax(slopes)
                                        critical_k = k_smooth[critical_idx]
                                        
                                        ax.axvline(x=critical_k, color='red', linestyle='--', 
                                                label=f'Critical K ≈ {critical_k:.4f}')
                                        ax.legend()
                                    
                                    st.pyplot(fig)
                                    
                                    # Additional information
                                    st.markdown(f"""
                                    <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                                        <p><b>Dataset:</b> {dataset_details['name']}</p>
                                        <p><b>Simulations:</b> {len(dataset_details['simulations'])}</p>
                                        <p><b>K Range:</b> {k_sorted.min():.4f} to {k_sorted.max():.4f}</p>
                                        <p><b>Max Synchronization:</b> {r_sorted.max():.4f}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("This dataset doesn't contain the required features for parameter sweep visualization.")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                    
                    elif vis_type == "Network Type Comparison":
                        st.markdown("<h4>Network Type Comparison</h4>", unsafe_allow_html=True)
                        
                        # Compare synchronization across different network types
                        try:
                            # Check if dataset has the required features
                            if any(f['name'] == 'network_type' for f in dataset_details['features']) and \
                               any(f['name'] == 'sync_trajectory' for f in dataset_details['features']):
                                
                                with st.spinner("Generating visualization..."):
                                    # Extract network types and synchronization trajectories
                                    for feature in dataset_details['features']:
                                        if feature['name'] == 'network_type':
                                            network_types = feature['data']['value']
                                        elif feature['name'] == 'sync_trajectory':
                                            sync_trajectories = feature['data']
                                    
                                    # Group by network type
                                    network_groups = {}
                                    for i, info in enumerate(network_types):
                                        net_type = info.get('network_type', 'unknown')
                                        if net_type not in network_groups:
                                            network_groups[net_type] = []
                                        network_groups[net_type].append(i)
                                    
                                    # Plot synchronization trajectories by network type
                                    fig, ax = plt.subplots(figsize=(10, 6))
                                    
                                    colors = {
                                        'all-to-all': '#1f77b4',  # blue
                                        'ring': '#ff7f0e',        # orange
                                        'random': '#2ca02c'       # green
                                    }
                                    
                                    # Plot mean and individual trajectories for each network type
                                    for net_type, indices in network_groups.items():
                                        # Extract trajectories for this network type
                                        trajectories = [sync_trajectories[idx] for idx in indices]
                                        
                                        # Compute the mean trajectory
                                        mean_traj = np.mean(trajectories, axis=0)
                                        std_traj = np.std(trajectories, axis=0)
                                        
                                        # Plot individual trajectories with low alpha
                                        for traj in trajectories:
                                            ax.plot(traj, color=colors.get(net_type, '#777777'), 
                                                   alpha=0.2, linewidth=1)
                                        
                                        # Plot the mean with error bands
                                        times = np.arange(len(mean_traj))
                                        ax.plot(times, mean_traj, color=colors.get(net_type, '#777777'), 
                                               linewidth=2, label=f'{net_type}')
                                        ax.fill_between(times, mean_traj - std_traj, mean_traj + std_traj, 
                                                      color=colors.get(net_type, '#777777'), alpha=0.2)
                                    
                                    # Format the plot
                                    ax.set_xlabel('Time Step', fontsize=12)
                                    ax.set_ylabel('Order Parameter (r)', fontsize=12)
                                    ax.set_title('Synchronization Dynamics by Network Type', fontsize=14)
                                    ax.legend()
                                    ax.grid(True, alpha=0.3)
                                    
                                    st.pyplot(fig)
                                    
                                    # Additional information
                                    st.markdown(f"""
                                    <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                                        <p><b>Dataset:</b> {dataset_details['name']}</p>
                                        <p><b>Network Types:</b> {', '.join(network_groups.keys())}</p>
                                        <p><b>Samples per Type:</b> {', '.join([f"{k}: {len(v)}" for k, v in network_groups.items()])}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("This dataset doesn't contain the required features for network type comparison.")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                    
                    elif vis_type == "Critical Coupling Analysis":
                        st.markdown("<h4>Critical Coupling Analysis</h4>", unsafe_allow_html=True)
                        
                        # Visualize critical coupling transition in detail
                        try:
                            # Check if dataset has the required features
                            if any(f['name'] == 'coupling_strength' for f in dataset_details['features']) and \
                               any(f['name'] == 'order_parameter_timeseries' for f in dataset_details['features']):
                                
                                with st.spinner("Generating visualization..."):
                                    # Extract features
                                    for feature in dataset_details['features']:
                                        if feature['name'] == 'coupling_strength':
                                            k_values = feature['data']['value']
                                        elif feature['name'] == 'order_parameter_timeseries':
                                            r_timeseries = feature['data']
                                        elif feature['name'] == 'steady_state_sync':
                                            r_final = feature['data']['value']
                                    
                                    # Create two subplots
                                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                                    
                                    # 1. Plot the final order parameter vs K
                                    sort_idx = np.argsort(k_values)
                                    k_sorted = k_values[sort_idx]
                                    r_final_sorted = r_final[sort_idx]
                                    
                                    ax1.scatter(k_sorted, r_final_sorted, c=k_sorted, cmap='viridis', s=50, alpha=0.8)
                                    ax1.plot(k_sorted, r_final_sorted, 'k--', alpha=0.3)
                                    
                                    # Add critical coupling estimation
                                    if len(k_sorted) > 5:
                                        # Smooth the curve
                                        k_smooth = np.linspace(k_sorted.min(), k_sorted.max(), 100)
                                        r_smooth = interp1d(k_sorted, r_final_sorted, kind='cubic')(k_smooth)
                                        
                                        # Find the point of steepest slope
                                        slopes = np.gradient(r_smooth, k_smooth)
                                        critical_idx = np.argmax(slopes)
                                        critical_k = k_smooth[critical_idx]
                                        
                                        ax1.axvline(x=critical_k, color='red', linestyle='--', 
                                                  label=f'Critical K ≈ {critical_k:.4f}')
                                        ax1.legend()
                                    
                                    ax1.set_xlabel('Coupling Strength (K)', fontsize=12)
                                    ax1.set_ylabel('Final Order Parameter (r)', fontsize=12)
                                    ax1.set_title('Synchronization Transition', fontsize=14)
                                    ax1.grid(True, alpha=0.3)
                                    
                                    # 2. Plot the order parameter dynamics for different K values
                                    # Select a subset of K values to visualize
                                    n_curves = min(8, len(k_values))
                                    k_indices = np.linspace(0, len(k_values)-1, n_curves, dtype=int)
                                    
                                    # Create a colormap
                                    cmap = cm.viridis
                                    
                                    # Plot each curve with color based on K value
                                    for i, idx in enumerate(k_indices):
                                        k = k_values[idx]
                                        r_curve = r_timeseries[idx]
                                        
                                        # Normalize color by position in K range
                                        norm_k = (k - k_values.min()) / (k_values.max() - k_values.min())
                                        color = cmap(norm_k)
                                        
                                        # Plot with label
                                        ax2.plot(r_curve, color=color, linewidth=2, 
                                               label=f'K = {k:.2f}')
                                    
                                    ax2.set_xlabel('Time Steps', fontsize=12)
                                    ax2.set_ylabel('Order Parameter (r)', fontsize=12)
                                    ax2.set_title('Synchronization Dynamics for Different K', fontsize=14)
                                    ax2.grid(True, alpha=0.3)
                                    
                                    # Add legend if not too many curves
                                    handles, labels = ax2.get_legend_handles_labels()
                                    if handles:
                                        ax2.legend(handles[:5], labels[:5], loc='lower right')
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Add a colormap showing the full range of K values
                                    st.markdown("<h5>Full Color Legend for K Values</h5>", unsafe_allow_html=True)
                                    
                                    # Create a separate figure for the colorbar
                                    fig_cbar, ax_cbar = plt.subplots(figsize=(10, 1))
                                    norm = plt.Normalize(k_values.min(), k_values.max())
                                    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), 
                                                   cax=ax_cbar, orientation='horizontal')
                                    cb.set_label('Coupling Strength (K)')
                                    
                                    st.pyplot(fig_cbar)
                                    
                                    # Additional information
                                    st.markdown(f"""
                                    <div style='background-color: rgba(0,0,0,0.2); padding: 10px; border-radius: 5px;'>
                                        <p><b>Dataset:</b> {dataset_details['name']}</p>
                                        <p><b>Simulations:</b> {len(dataset_details['simulations'])}</p>
                                        <p><b>K Range:</b> {k_values.min():.4f} to {k_values.max():.4f}</p>
                                        <p><b>Estimated Critical K:</b> {critical_k if 'critical_k' in locals() else 'Not available'}</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("This dataset doesn't contain the required features for critical coupling analysis.")
                        except Exception as e:
                            st.error(f"Error generating visualization: {str(e)}")
                else:
                    st.error("Could not load dataset details. Please try refreshing the datasets list.")