"""
Frequency distribution handling for the Kuramoto model simulator.
"""

import streamlit as st
import numpy as np

def render_frequency_controls(freq_type, n_oscillators):
    """
    Render UI controls for frequency distribution and generate frequencies.
    
    Args:
        freq_type: Type of frequency distribution
        n_oscillators: Number of oscillators
        
    Returns:
        ndarray: Array of frequencies based on selected distribution
    """
    # Parameters for frequency distribution
    if freq_type == "Normal":
        return _render_normal_distribution(n_oscillators)
    elif freq_type == "Uniform":
        return _render_uniform_distribution(n_oscillators)
    elif freq_type == "Bimodal":
        return _render_bimodal_distribution(n_oscillators)
    elif freq_type == "Golden Ratio":
        return _render_golden_ratio_distribution(n_oscillators)
    else:  # Custom
        return _render_custom_distribution(n_oscillators)

def _render_normal_distribution(n_oscillators):
    """Render normal distribution controls and generate frequencies."""
    freq_mean = st.sidebar.slider("Mean", -2.0, 2.0, step=0.1, key="freq_mean")
    freq_std = st.sidebar.slider("Standard Deviation", 0.1, 3.0, step=0.1, key="freq_std")
    return np.random.normal(freq_mean, freq_std, n_oscillators)

def _render_uniform_distribution(n_oscillators):
    """Render uniform distribution controls and generate frequencies."""
    freq_min = st.sidebar.slider("Minimum", -5.0, 0.0, step=0.1, key="freq_min")
    freq_max = st.sidebar.slider("Maximum", 0.0, 5.0, step=0.1, key="freq_max")
    return np.random.uniform(freq_min, freq_max, n_oscillators)

def _render_bimodal_distribution(n_oscillators):
    """Render bimodal distribution controls and generate frequencies."""
    peak1 = st.sidebar.slider("Peak 1", -5.0, 0.0, step=0.1, key="peak1")
    peak2 = st.sidebar.slider("Peak 2", 0.0, 5.0, step=0.1, key="peak2")
    mix = np.random.choice([0, 1], size=n_oscillators)
    freq1 = np.random.normal(peak1, 0.3, n_oscillators)
    freq2 = np.random.normal(peak2, 0.3, n_oscillators)
    return mix * freq1 + (1 - mix) * freq2

def _render_golden_ratio_distribution(n_oscillators):
    """Render golden ratio distribution and generate frequencies."""
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
    return np.array([golden_ratio_start + i * phi for i in range(n_oscillators)])

def _render_custom_distribution(n_oscillators):
    """Render custom distribution controls and generate frequencies."""
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
        return frequencies
    except:
        st.sidebar.error("Invalid frequency input. Using normal distribution instead.")
        return np.random.normal(0, 1, n_oscillators)