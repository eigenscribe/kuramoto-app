import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from io import BytesIO
import base64
from kuramoto_model import KuramotoModel
import time

# Set page title and configuration
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Title and description
st.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)

st.markdown("""
<div class='section'>
    <h2 class='gradient_text2'>About the Kuramoto Model</h2>
    <div class='section-content'>
        <p>The Kuramoto model is a mathematical model used to describe synchronization in systems of coupled oscillators.</p>
        <p>It was first introduced by Yoshiki Kuramoto in 1975 and has applications in neuroscience, chemical oscillators, 
        power grids, and many other complex systems.</p>
        <p>The model describes each oscillator by its phase θ, where the dynamics are governed by:</p>
        <p style='text-align: center; font-size: 1.2em;'>
            dθ<sub>i</sub>/dt = ω<sub>i</sub> + (K/N) Σ<sub>j=1</sub><sup>N</sup> sin(θ<sub>j</sub> - θ<sub>i</sub>)
        </p>
        <p>where:</p>
        <ul>
            <li>θ<sub>i</sub> is the phase of oscillator i</li>
            <li>ω<sub>i</sub> is the natural frequency of oscillator i</li>
            <li>K is the coupling strength between oscillators</li>
            <li>N is the total number of oscillators</li>
        </ul>
        <p>The order parameter r(t) measures the coherence of the system, with r = 1 indicating complete synchronization.</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Create sidebar with parameters
st.sidebar.markdown("<h2 class='gradient_text2'>Simulation Parameters</h2>", unsafe_allow_html=True)

# Number of oscillators slider
n_oscillators = st.sidebar.slider(
    "Number of Oscillators",
    min_value=2,
    max_value=50,
    value=10,
    step=1,
    help="Number of oscillators in the system"
)

# Coupling strength slider
coupling_strength = st.sidebar.slider(
    "Coupling Strength (K)",
    min_value=0.0,
    max_value=10.0,
    value=1.0,
    step=0.1,
    help="Strength of coupling between oscillators"
)

# Frequency distribution type
freq_type = st.sidebar.selectbox(
    "Frequency Distribution",
    ["Normal", "Uniform", "Bimodal", "Custom"],
    help="Distribution of natural frequencies"
)

# Parameters for frequency distribution
if freq_type == "Normal":
    freq_mean = st.sidebar.slider("Mean", -2.0, 2.0, 0.0, 0.1)
    freq_std = st.sidebar.slider("Standard Deviation", 0.1, 3.0, 1.0, 0.1)
    frequencies = np.random.normal(freq_mean, freq_std, n_oscillators)
    
elif freq_type == "Uniform":
    freq_min = st.sidebar.slider("Minimum", -5.0, 0.0, -1.0, 0.1)
    freq_max = st.sidebar.slider("Maximum", 0.0, 5.0, 1.0, 0.1)
    frequencies = np.random.uniform(freq_min, freq_max, n_oscillators)
    
elif freq_type == "Bimodal":
    peak1 = st.sidebar.slider("Peak 1", -5.0, 0.0, -1.0, 0.1)
    peak2 = st.sidebar.slider("Peak 2", 0.0, 5.0, 1.0, 0.1)
    mix = np.random.choice([0, 1], size=n_oscillators)
    frequencies = mix * np.random.normal(peak1, 0.3, n_oscillators) + (1 - mix) * np.random.normal(peak2, 0.3, n_oscillators)
    
else:  # Custom
    custom_freqs = st.sidebar.text_area(
        "Enter custom frequencies (comma-separated)",
        "0.5, 1.0, 1.5, 2.0, 2.5, 3.0, -0.5, -1.0, -1.5, -2.0"
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
    max_value=50.0,
    value=20.0,
    step=1.0,
    help="Total simulation time"
)

time_step = st.sidebar.slider(
    "Time Step",
    min_value=0.01,
    max_value=0.1,
    value=0.05,
    step=0.01,
    help="Time step for simulation"
)

# Initialize model with specified parameters
random_seed = st.sidebar.number_input("Random Seed", value=42, help="Seed for reproducibility")

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Simulation Results", "Animated Visualization", "Phase Distribution"])

# Function to simulate model
@st.cache_data(ttl=300)
def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step, random_seed):
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed
    )
    
    times, phases, order_parameter = model.simulate()
    return model, times, phases, order_parameter

# Run the simulation
model, times, phases, order_parameter = run_simulation(
    n_oscillators=n_oscillators,
    coupling_strength=coupling_strength,
    frequencies=frequencies,
    simulation_time=simulation_time,
    time_step=time_step,
    random_seed=random_seed
)

# Tab 1: Simulation Results
with tab1:
    st.markdown("<h2 class='gradient_text2'>Simulation Results</h2>", unsafe_allow_html=True)
    
    # Plot order parameter
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times, order_parameter)
    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Order Parameter r(t)', fontsize=12)
    ax1.set_title('Phase Synchronization in Kuramoto Model', fontsize=14)
    ax1.grid(True)
    ax1.set_ylim(0, 1.05)
    
    st.pyplot(fig1)
    
    st.markdown("""
    <div class='section'>
        <h3>Order Parameter Analysis</h3>
        <p>The order parameter r(t) measures the degree of synchronization among oscillators:</p>
        <ul>
            <li>r = 1: Complete synchronization (all oscillators have the same phase)</li>
            <li>r = 0: Complete desynchronization (phases are uniformly distributed)</li>
        </ul>
        <p>At critical coupling strength (K_c), the system transitions from desynchronized to partially synchronized state.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plot phases
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for i in range(n_oscillators):
        ax2.plot(times, phases[i, :] % (2 * np.pi), alpha=0.7)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Phase (mod 2π)', fontsize=12)
    ax2.set_title('Oscillator Phases Over Time', fontsize=14)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.grid(True)
    
    st.pyplot(fig2)
    
    st.markdown("""
    <div class='section'>
        <h3>Phase Evolution</h3>
        <p>The plot above shows how the phase of each oscillator evolves over time.</p>
        <p>When synchronized, oscillators will move with similar phases (lines will cluster together).</p>
    </div>
    """, unsafe_allow_html=True)

# Tab 2: Animated Visualization
with tab2:
    st.markdown("<h2 class='gradient_text2'>Animated Visualization</h2>", unsafe_allow_html=True)
    
    # Create oscillator visualization at different time points
    st.markdown("### Oscillators on Unit Circle")
    
    time_index = st.slider(
        "Time Index",
        min_value=0,
        max_value=len(times)-1,
        value=0,
        format="t = %.2f" % times[0]
    )
    
    # Display time and order parameter
    current_time = times[time_index]
    current_r = order_parameter[time_index]
    
    st.markdown(f"**Time:** {current_time:.2f}  |  **Order Parameter:** {current_r:.3f}")
    
    # Create visualization
    fig_circle = plt.figure(figsize=(8, 8))
    ax_circle = fig_circle.add_subplot(111)
    
    # Draw unit circle
    circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
    ax_circle.add_patch(circle)
    
    # Plot oscillators
    phases_at_time = phases[:, time_index]
    x = np.cos(phases_at_time)
    y = np.sin(phases_at_time)
    
    # Color oscillators by their natural frequency
    sc = ax_circle.scatter(x, y, c=frequencies, cmap='viridis', s=100, zorder=10)
    plt.colorbar(sc, ax=ax_circle, label='Natural Frequency')
    
    # Calculate and show order parameter
    r = order_parameter[time_index]
    psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
    
    # Draw arrow showing mean field
    ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                   head_width=0.05, head_length=0.1, fc='red', ec='red', 
                   width=0.02, zorder=5)
    
    ax_circle.set_xlim(-1.1, 1.1)
    ax_circle.set_ylim(-1.1, 1.1)
    ax_circle.set_aspect('equal')
    ax_circle.grid(True)
    ax_circle.set_title(f'Oscillators at time t={times[time_index]:.2f}, r={r:.3f}')
    
    st.pyplot(fig_circle)
    
    st.markdown("""
    <div class='section'>
        <h3>Interactive Animation</h3>
        <p>Drag the slider to see how oscillators evolve over time.</p>
        <p>Each dot represents an oscillator, with color indicating its natural frequency.</p>
        <p>The red arrow shows the mean field vector, with length equal to the order parameter r.</p>
    </div>
    """, unsafe_allow_html=True)

# Tab 3: Phase Distribution
with tab3:
    st.markdown("<h2 class='gradient_text2'>Frequency and Phase Distributions</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Plot natural frequency distribution
        fig_freq, ax_freq = plt.subplots(figsize=(6, 4))
        ax_freq.hist(frequencies, bins=15, alpha=0.7, color='blue')
        ax_freq.set_xlabel('Natural Frequency (ω)', fontsize=12)
        ax_freq.set_ylabel('Count', fontsize=12)
        ax_freq.set_title('Natural Frequency Distribution', fontsize=14)
        ax_freq.grid(True, alpha=0.3)
        st.pyplot(fig_freq)
        
        st.markdown("""
        <div class='section'>
            <h3>Natural Frequencies</h3>
            <p>The distribution of natural frequencies affects how easily oscillators synchronize.</p>
            <p>A wider distribution requires stronger coupling to achieve synchronization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Plot phase histogram at selected time
        time_idx_hist = st.slider(
            "Select Time Index for Phase Histogram",
            min_value=0,
            max_value=len(times)-1,
            value=len(times)//2,
            format="t = %.2f" % times[len(times)//2]
        )
        
        fig_phase, ax_phase = plt.subplots(figsize=(6, 4))
        
        phases_at_t = phases[:, time_idx_hist] % (2 * np.pi)
        ax_phase.hist(phases_at_t, bins=15, alpha=0.7, color='green')
        ax_phase.set_xlabel('Phase (mod 2π)', fontsize=12)
        ax_phase.set_ylabel('Count', fontsize=12)
        ax_phase.set_title(f'Phase Distribution at t={times[time_idx_hist]:.2f}', fontsize=14)
        ax_phase.grid(True, alpha=0.3)
        
        st.pyplot(fig_phase)
        
        # Show order parameter at this time
        r_at_t = order_parameter[time_idx_hist]
        
        if r_at_t > 0.8:
            sync_status = "Strong synchronization"
        elif r_at_t > 0.5:
            sync_status = "Partial synchronization"
        else:
            sync_status = "Weak/no synchronization"
            
        st.markdown(f"""
        <div class='section'>
            <h3>Phase Distribution Analysis</h3>
            <p>Order parameter at this time: <b>{r_at_t:.3f}</b></p>
            <p>Status: <b>{sync_status}</b></p>
            <p>When synchronized, phases cluster together. When desynchronized, phases spread uniformly.</p>
        </div>
        """, unsafe_allow_html=True)

# Display info about Kuramoto model applications
st.markdown("""
<div class='section'>
    <h2 class='gradient_text2'>Applications of the Kuramoto Model</h2>
    <div class='section-content'>
        <ul>
            <li><b>Neuroscience</b>: Modeling neural oscillations and brain rhythms</li>
            <li><b>Power Grids</b>: Synchronization of power generators</li>
            <li><b>Biology</b>: Circadian rhythms and firefly synchronization</li>
            <li><b>Chemistry</b>: Coupled chemical oscillators</li>
            <li><b>Crowd Dynamics</b>: Synchronized clapping after performances</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Add references
st.markdown("""
<div class='section'>
    <h2 class='gradient_text2'>References</h2>
    <div class='section-content'>
        <ul>
            <li>Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.</li>
            <li>Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators.</li>
            <li>Acebrón, J. A., et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena.</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8em;'>
    <p>Kuramoto Model Simulator © 2023 | Interactive Application</p>
</div>
""", unsafe_allow_html=True)
