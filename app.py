import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from io import BytesIO
import base64
from kuramoto_model import KuramotoModel
import time

# Set up Matplotlib style for dark theme plots
plt.style.use('dark_background')
plt.rcParams.update({
    'axes.facecolor': '#1e1e1e',
    'figure.facecolor': '#1e1e1e',
    'savefig.facecolor': '#1e1e1e',
    'axes.edgecolor': '#757575',
    'axes.labelcolor': 'white',
    'axes.grid': True,
    'grid.color': '#333333',
    'grid.linestyle': '--',
    'grid.alpha': 0.7,
    'xtick.color': 'white',
    'ytick.color': 'white',
    'text.color': 'white',
    'figure.figsize': (10, 6),
    'font.size': 12,
    'lines.linewidth': 2,
})

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

# Add custom background and custom font
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Aclonica&display=swap');
    
    .stApp {
        background-image: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.7)), 
                         url('static/images/wisp.jpg');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Ensure Aclonica font is applied everywhere */
    body, div, p, h1, h2, h3, h4, h5, h6, li, span, label, button {
        font-family: 'Aclonica', sans-serif !important;
    }
    
    /* Fix Streamlit buttons to use Aclonica */
    button, .stButton button, .stDownloadButton button {
        font-family: 'Aclonica', sans-serif !important;
    }
    
    /* Fix Streamlit widgets text */
    .stSlider label, .stSelectbox label, .stNumberInput label {
        font-family: 'Aclonica', sans-serif !important;
    }
</style>
""", unsafe_allow_html=True)

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
    
    # Enhanced order parameter plot
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    # Add background gradient
    ax1.set_facecolor('#1a1a1a')
    
    # Add subtle horizontal bands for visual reference
    for y in np.linspace(0, 1, 6):
        ax1.axhspan(y-0.05, y+0.05, color='#222233', alpha=0.3, zorder=0)
    
    # Plot order parameter with gradient line effect
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    
    # Create points and segments
    points = np.array([times, order_parameter]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Create a custom colormap that uses our primary gradients
    cmap = LinearSegmentedColormap.from_list("order_param", 
                                           ["#00ffee", "#27aaff", "#14a5ff", "#8138ff"], 
                                           N=256)
    
    # Create the line collection with gradient coloring based on order parameter value
    lc = LineCollection(segments, cmap=cmap, linewidth=3, zorder=5)
    lc.set_array(order_parameter)
    ax1.add_collection(lc)
    
    # Add highlights at important thresholds
    ax1.axhline(y=0.5, color='#aaaaaa', linestyle='--', alpha=0.5, zorder=1, 
               label='Partial Synchronization (r=0.5)')
    ax1.axhline(y=0.8, color='#ffffff', linestyle='--', alpha=0.5, zorder=1,
               label='Strong Synchronization (r=0.8)')
    
    # Enhance the plot appearance
    ax1.set_xlim(times.min(), times.max())
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
    ax1.set_ylabel('Order Parameter r(t)', fontsize=13, fontweight='bold', color='white')
    ax1.set_title('Phase Synchronization in Kuramoto Model', 
                 fontsize=15, fontweight='bold', color='white', pad=15)
    
    # Create custom grid
    ax1.grid(True, color='#333333', alpha=0.5, linestyle=':')
    
    # Add legend
    ax1.legend(loc='upper right', framealpha=0.7)
    
    # Add subtle box around the plot
    for spine in ax1.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1)
    
    # Add explanation annotations
    if max(order_parameter) > 0.8:
        max_idx = np.argmax(order_parameter)
        ax1.annotate('Peak synchronization', 
                   xy=(times[max_idx], order_parameter[max_idx]),
                   xytext=(times[max_idx]-1, order_parameter[max_idx]+0.15),
                   fontsize=11,
                   color='white',
                   arrowprops=dict(facecolor='white', shrink=0.05, width=1.5, alpha=0.7))
    
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
    
    # Enhanced phase plot
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Add background and styling
    ax2.set_facecolor('#1a1a1a')
    
    # Create transparent bands at phase regions
    for y in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
        ax2.axhspan(y-0.1, y+0.1, color='#222233', alpha=0.4, zorder=0)
    
    # Create custom colormap that matches our gradient theme
    custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                 ["#00ffee", "#27aaff", "#14a5ff", "#8138ff"], 
                                                 N=256)
    
    # Plot phases with color gradient matching our theme
    line_colors = custom_cmap(np.linspace(0, 1, n_oscillators))
    for i in range(n_oscillators):
        ax2.plot(times, phases[i, :] % (2 * np.pi), color=line_colors[i], alpha=0.8, linewidth=1.5)
    
    # Add labels for key phase positions
    phase_labels = [(0, '0'), (np.pi/2, 'π/2'), (np.pi, 'π'), (3*np.pi/2, '3π/2'), (2*np.pi, '2π')]
    for y, label in phase_labels:
        ax2.annotate(label, xy=(-0.5, y), xycoords=('axes fraction', 'data'),
                   fontsize=11, color='white', ha='center', va='center')
    
    # Plot styling
    ax2.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
    ax2.set_ylabel('Phase (mod 2π)', fontsize=13, fontweight='bold', color='white')
    ax2.set_title('Oscillator Phases Over Time', fontsize=15, fontweight='bold', color='white', pad=15)
    ax2.set_ylim(0, 2 * np.pi)
    ax2.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax2.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    
    # Custom grid
    ax2.grid(True, color='#333333', alpha=0.4, linestyle=':')
    
    # Add box around the plot
    for spine in ax2.spines.values():
        spine.set_edgecolor('#555555')
        spine.set_linewidth(1)
    
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
    
    # Create visualization with enhanced visuals for dark theme
    fig_circle = plt.figure(figsize=(8, 8))
    ax_circle = fig_circle.add_subplot(111)
    
    # Add background glow effect
    bg_circle = plt.Circle((0, 0), 1.1, fill=True, color='#121a24', alpha=0.6, zorder=1)
    ax_circle.add_patch(bg_circle)
    
    # Add subtle circle rings for reference
    for radius in [0.25, 0.5, 0.75]:
        ring = plt.Circle((0, 0), radius, fill=False, color='#334455', 
                         linestyle=':', alpha=0.5, zorder=2)
        ax_circle.add_patch(ring)
    
    # Draw unit circle with glow effect
    circle_glow = plt.Circle((0, 0), 1.02, fill=False, color='#4488aa', alpha=0.3, linewidth=3, zorder=3)
    ax_circle.add_patch(circle_glow)
    
    circle = plt.Circle((0, 0), 1, fill=False, color='#66ccff', linestyle='-', 
                       linewidth=1.5, alpha=0.8, zorder=4)
    ax_circle.add_patch(circle)
    
    # Plot oscillators
    phases_at_time = phases[:, time_index]
    x = np.cos(phases_at_time)
    y = np.sin(phases_at_time)
    
    # Create custom colormap that matches our gradient theme
    custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                 ["#00ffee", "#27aaff", "#14a5ff", "#8138ff"], 
                                                 N=256)
    
    # Color oscillators by their natural frequency with enhanced visuals
    sc = ax_circle.scatter(x, y, c=frequencies, cmap=custom_cmap, s=120, 
                          alpha=0.9, edgecolor='white', linewidth=1, zorder=10)
    cbar = plt.colorbar(sc, ax=ax_circle, label='Natural Frequency')
    cbar.ax.yaxis.label.set_color('white')
    
    # Calculate and show order parameter
    r = order_parameter[time_index]
    psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
    
    # Draw arrow showing mean field with glow effect
    # First add glow/shadow
    ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                   head_width=0.07, head_length=0.12, fc='#ff6655', ec='#ff6655', 
                   width=0.03, alpha=0.3, zorder=5)
    
    # Then add main arrow
    ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                   head_width=0.05, head_length=0.1, fc='#ff3322', ec='#ff3322', 
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
    ax_circle.set_title(f'Oscillators at time t={times[time_index]:.2f}, r={r:.3f}', 
                       color='white', fontsize=14, pad=15)
    
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
    st.markdown("<h2 class='gradient_text2'>Phase Distribution Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
        <p>The phase distribution shows how oscillator phases are distributed at each moment in time.</p>
        <p>When oscillators synchronize, their phases cluster together, resulting in peaks in the histogram.</p>
        <p>Remember that natural frequencies are constant intrinsic properties of each oscillator, while phases evolve over time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Plot phase histogram at selected time
    time_idx_hist = st.slider(
        "Select Time for Phase Distribution",
        min_value=0,
        max_value=len(times)-1,
        value=len(times)//2,
        format="t = %.2f" % times[len(times)//2]
    )
    
    # Two columns for phase histogram and analysis
    col1, col2 = st.columns([3, 2])
    
    with col1:
        fig_phase, ax_phase = plt.subplots(figsize=(8, 4))
        
        phases_at_t = phases[:, time_idx_hist] % (2 * np.pi)
        
        # Create a beautiful gradient for the histogram
        # Use a gradient colormap for the histogram
        n_bins = 15
        counts, bin_edges = np.histogram(phases_at_t, bins=n_bins)
        
        # Create custom colormap that matches our gradient theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                  ["#00ffee", "#27aaff", "#14a5ff", "#8138ff"], 
                                                  N=256)
        
        # Create custom colors with a gradient effect that matches our theme
        colors = custom_cmap(np.linspace(0.1, 0.9, n_bins))
        
        # Plot the histogram with gradient colors and outline
        bars = ax_phase.bar(
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
            ax_phase.add_patch(glow)
        
        # Enhance the axes and labels
        ax_phase.set_facecolor('#1a1a1a')
        ax_phase.set_xlabel('Phase (mod 2π)', fontsize=12, fontweight='bold', color='white')
        ax_phase.set_ylabel('Count', fontsize=12, fontweight='bold', color='white')
        ax_phase.set_title(f'Phase Distribution at t={times[time_idx_hist]:.2f}', 
                          fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Highlight synchronized phases with vertical line if order is high
        r_at_t = order_parameter[time_idx_hist]
        if r_at_t > 0.6:
            psi = np.angle(np.sum(np.exp(1j * phases_at_t))) % (2 * np.pi)
            ax_phase.axvline(x=psi, color='#ff5555', linestyle='-', linewidth=2, alpha=0.7,
                           label=f'Mean Phase ψ={psi:.2f}')
            ax_phase.legend(framealpha=0.7)
        
        # Customize grid
        ax_phase.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Add a subtle box around the plot
        for spine in ax_phase.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        st.pyplot(fig_phase)
    
    with col2:
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
            <p>Order parameter at t={times[time_idx_hist]:.2f}: <b>{r_at_t:.3f}</b></p>
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
