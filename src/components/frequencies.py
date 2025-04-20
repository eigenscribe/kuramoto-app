"""
Frequencies component for the Kuramoto simulator.
This module provides functions for generating and visualizing frequency distributions.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import pandas as pd

def generate_frequencies(distribution_type, n_oscillators, params=None, random_seed=None):
    """
    Generate natural frequencies based on the specified distribution.
    
    Parameters:
    -----------
    distribution_type : str
        Type of distribution ("Normal", "Uniform", "Bimodal", "Custom")
    n_oscillators : int
        Number of oscillators
    params : dict, optional
        Parameters for the distribution
    random_seed : int, optional
        Random seed for reproducibility
    
    Returns:
    --------
    ndarray
        Array of natural frequencies
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    if distribution_type == "Normal":
        # Normal distribution
        mean = params.get('mean', 0.0) if params else 0.0
        std = params.get('std', 1.0) if params else 1.0
        return np.random.normal(mean, std, n_oscillators)
        
    elif distribution_type == "Uniform":
        # Uniform distribution
        min_val = params.get('min', -1.0) if params else -1.0
        max_val = params.get('max', 1.0) if params else 1.0
        return np.random.uniform(min_val, max_val, n_oscillators)
        
    elif distribution_type == "Bimodal":
        # Bimodal distribution (mixture of two normals)
        peak1 = params.get('peak1', -1.0) if params else -1.0
        peak2 = params.get('peak2', 1.0) if params else 1.0
        std = 0.3  # Fixed standard deviation for each peak
        
        # Generate half from each peak
        n1 = n_oscillators // 2
        n2 = n_oscillators - n1
        
        freqs1 = np.random.normal(peak1, std, n1)
        freqs2 = np.random.normal(peak2, std, n2)
        
        return np.concatenate([freqs1, freqs2])
        
    elif distribution_type == "Custom":
        # Custom frequencies provided by the user
        custom_freqs = params.get('custom_freqs', None) if params else None
        
        if custom_freqs is None or len(custom_freqs) != n_oscillators:
            # If no custom frequencies or wrong length, fall back to normal distribution
            return np.random.normal(0, 1, n_oscillators)
        else:
            return np.array(custom_freqs)
    else:
        # Default to normal distribution
        return np.random.normal(0, 1, n_oscillators)

def plot_frequency_histogram(frequencies, distribution_type=None, params=None):
    """
    Plot a histogram of oscillator frequencies.
    
    Parameters:
    -----------
    frequencies : ndarray
        Array of oscillator frequencies
    distribution_type : str, optional
        Type of distribution
    params : dict, optional
        Parameters of the distribution
    
    Returns:
    --------
    str
        Base64 encoded PNG image of the histogram
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram
    bins = min(20, len(frequencies))
    n, bins, patches = ax.hist(frequencies, bins=bins, alpha=0.7, color='#3498db', density=True)
    
    # Calculate mean and standard deviation
    mean = np.mean(frequencies)
    std = np.std(frequencies)
    
    # Get range for theoretical distribution
    x = np.linspace(min(frequencies) - 1, max(frequencies) + 1, 1000)
    
    # Plot theoretical distribution if parameters are available
    if distribution_type and params:
        if distribution_type == "Normal":
            # Normal distribution
            pdf = (1 / (params['std'] * np.sqrt(2 * np.pi))) * np.exp(-(x - params['mean'])**2 / (2 * params['std']**2))
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Normal(μ={params["mean"]:.2f}, σ={params["std"]:.2f})')
        elif distribution_type == "Uniform":
            # Uniform distribution
            pdf = np.zeros_like(x)
            mask = (x >= params['min']) & (x <= params['max'])
            pdf[mask] = 1.0 / (params['max'] - params['min'])
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Uniform({params["min"]:.2f}, {params["max"]:.2f})')
        elif distribution_type == "Bimodal":
            # Bimodal distribution (mixture of two normals)
            std = 0.3  # Fixed std for bimodal
            pdf1 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - params['peak1'])**2 / (2 * std**2))
            pdf2 = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-(x - params['peak2'])**2 / (2 * std**2))
            pdf = 0.5 * (pdf1 + pdf2)
            ax.plot(x, pdf, 'r-', linewidth=2, label=f'Bimodal({params["peak1"]:.2f}, {params["peak2"]:.2f})')
    
    # Add labels and title
    ax.set_xlabel('Natural Frequency (ω)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.set_title(f'Natural Frequency Distribution', fontsize=14)
    
    # Add text with summary statistics
    stats_text = f'Mean: {mean:.3f}\nStd Dev: {std:.3f}\nMin: {min(frequencies):.3f}\nMax: {max(frequencies):.3f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add legend if theoretical distribution is plotted
    if distribution_type and params:
        ax.legend()
    
    # Add grid
    ax.grid(linestyle=':', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert figure to base64 image
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_str = base64.b64encode(buf.getvalue()).decode()
    plt.close(fig)
    
    return img_str

def render_frequencies_tab(frequency_params):
    """
    Render the frequencies tab with frequency distribution options and visualization.
    
    Parameters:
    -----------
    frequency_params : dict
        Parameters for frequency generation
    
    Returns:
    --------
    ndarray
        Array of frequencies to use for simulation
    """
    st.markdown("## 🌊 Oscillator Frequencies")
    
    distribution_type = frequency_params.get('frequency_distribution', 'Normal')
    n_oscillators = frequency_params.get('n_oscillators', 10)
    random_seed = frequency_params.get('random_seed', 42)
    
    st.markdown(f"### Frequency Distribution: {distribution_type}")
    
    # Initialize frequencies array
    frequencies = None
    
    if distribution_type == "Custom":
        st.markdown("#### Define Custom Frequencies")
        st.markdown("Upload a CSV file with frequencies or enter values manually.")
        
        input_method = st.radio(
            "Choose input method:",
            options=["Upload CSV", "Manual Entry", "Interactive Input"]
        )
        
        if input_method == "Upload CSV":
            uploaded_file = st.file_uploader("Upload frequencies CSV", type=['csv', 'txt'])
            
            if uploaded_file is not None:
                try:
                    # Load frequencies from file
                    df = pd.read_csv(uploaded_file, header=None)
                    custom_freqs = df.values.flatten()
                    
                    # Validate length
                    if len(custom_freqs) != n_oscillators:
                        st.warning(f"Number of frequencies ({len(custom_freqs)}) doesn't match the number of oscillators ({n_oscillators}). Frequencies will be adjusted.")
                        
                        if len(custom_freqs) > n_oscillators:
                            # Truncate if more frequencies provided
                            custom_freqs = custom_freqs[:n_oscillators]
                        else:
                            # Repeat values if fewer frequencies provided
                            repeats = int(np.ceil(n_oscillators / len(custom_freqs)))
                            custom_freqs = np.tile(custom_freqs, repeats)[:n_oscillators]
                    
                    st.success("Custom frequencies loaded successfully.")
                    frequencies = np.array(custom_freqs)
                except Exception as e:
                    st.error(f"Error loading frequencies: {str(e)}")
        
        elif input_method == "Manual Entry":
            freq_text = st.text_area(
                f"Enter {n_oscillators} comma-separated frequency values:",
                height=100,
                help="Example: 0.5, 1.2, -0.8, ..."
            )
            
            if freq_text:
                try:
                    # Parse frequencies from text
                    values = [float(val.strip()) for val in freq_text.split(',')]
                    
                    # Validate length
                    if len(values) != n_oscillators:
                        st.warning(f"Number of frequencies ({len(values)}) doesn't match the number of oscillators ({n_oscillators}). Frequencies will be adjusted.")
                        
                        if len(values) > n_oscillators:
                            # Truncate if more frequencies provided
                            values = values[:n_oscillators]
                        else:
                            # Repeat values if fewer frequencies provided
                            repeats = int(np.ceil(n_oscillators / len(values)))
                            values = np.tile(values, repeats)[:n_oscillators]
                    
                    st.success("Custom frequencies created successfully.")
                    frequencies = np.array(values)
                except Exception as e:
                    st.error(f"Error parsing frequencies: {str(e)}")
        
        elif input_method == "Interactive Input":
            st.markdown(f"Adjust each oscillator's frequency individually:")
            
            # Create sliders for each frequency, but limit to a reasonable number
            max_sliders = min(n_oscillators, 20)
            if n_oscillators > max_sliders:
                st.warning(f"Showing {max_sliders} sliders out of {n_oscillators} oscillators. Others will use generated values.")
            
            custom_freqs = []
            
            # Use columns to make the UI more compact
            cols_per_row = 2
            rows = (max_sliders + cols_per_row - 1) // cols_per_row
            
            for i in range(rows):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    idx = i * cols_per_row + j
                    if idx < max_sliders:
                        with cols[j]:
                            freq = st.slider(f"Oscillator {idx+1}", -5.0, 5.0, 0.0, step=0.1, key=f"freq_{idx}")
                            custom_freqs.append(freq)
            
            # Generate remaining frequencies randomly if needed
            if n_oscillators > max_sliders:
                remaining = n_oscillators - max_sliders
                np.random.seed(random_seed)
                additional_freqs = np.random.normal(0, 1, remaining)
                custom_freqs.extend(additional_freqs)
            
            frequencies = np.array(custom_freqs)
            
            # Let the user set a common pattern
            st.markdown("### Quick Patterns")
            pattern_type = st.selectbox(
                "Apply a pattern to all frequencies:",
                options=["None", "Linear", "Quadratic", "Sine Wave", "Cosine Wave", "Alternating"]
            )
            
            if pattern_type != "None":
                amplitude = st.slider("Pattern Amplitude", 0.1, 5.0, 1.0, step=0.1)
                offset = st.slider("Pattern Offset", -3.0, 3.0, 0.0, step=0.1)
                
                # Apply the selected pattern
                x = np.linspace(0, 1, n_oscillators)
                
                if pattern_type == "Linear":
                    pattern = offset + amplitude * x
                elif pattern_type == "Quadratic":
                    pattern = offset + amplitude * (x - 0.5)**2
                elif pattern_type == "Sine Wave":
                    pattern = offset + amplitude * np.sin(2 * np.pi * x)
                elif pattern_type == "Cosine Wave":
                    pattern = offset + amplitude * np.cos(2 * np.pi * x)
                elif pattern_type == "Alternating":
                    pattern = offset + amplitude * np.array([1 if i % 2 == 0 else -1 for i in range(n_oscillators)])
                
                # Apply pattern if button is clicked
                if st.button("Apply Pattern"):
                    frequencies = pattern
    
    # Generate frequencies based on distribution if not manually specified
    if frequencies is None:
        # Extract parameters based on distribution type
        dist_params = frequency_params.get('freq_params', {})
        
        # Generate frequencies
        frequencies = generate_frequencies(distribution_type, n_oscillators, dist_params, random_seed)
    
    # Display frequencies
    st.markdown("### Frequency Distribution")
    img_str = plot_frequency_histogram(frequencies, distribution_type, frequency_params.get('freq_params', {}))
    st.image(f"data:image/png;base64,{img_str}", use_column_width=True)
    
    # Display frequency statistics
    st.markdown("### Frequency Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Frequency", f"{np.mean(frequencies):.3f}")
        st.metric("Median Frequency", f"{np.median(frequencies):.3f}")
    
    with col2:
        st.metric("Frequency Std Dev", f"{np.std(frequencies):.3f}")
        st.metric("Frequency Range", f"{max(frequencies) - min(frequencies):.3f}")
    
    with col3:
        st.metric("Minimum Frequency", f"{min(frequencies):.3f}")
        st.metric("Maximum Frequency", f"{max(frequencies):.3f}")
    
    # Option to use these frequencies
    use_freqs = st.checkbox("Use these frequencies for simulation", value=True)
    
    if use_freqs:
        return frequencies
    else:
        return None