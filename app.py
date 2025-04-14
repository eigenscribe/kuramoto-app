import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
from kuramoto_model import KuramotoModel
import time
from database import store_simulation, get_simulation, list_simulations, delete_simulation

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

# Get the base64 encoded image
import base64
with open("wisp.base64", "r") as f:
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

# Title
st.markdown("<h1 class='gradient_text1'>Kuramoto Model Simulator</h1>", unsafe_allow_html=True)

# Create sidebar with parameters
st.sidebar.markdown("<h2 class='gradient_text1'>Simulation Parameters</h2>", unsafe_allow_html=True)

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

# Network Connectivity Configuration
st.sidebar.markdown("<h3 class='gradient_text1'>Network Connectivity</h3>", unsafe_allow_html=True)
network_type = st.sidebar.radio(
    "Network Type",
    options=["Fully Connected", "Custom Adjacency Matrix"],
    help="Define how oscillators are connected to each other"
)

# Custom adjacency matrix input
adj_matrix = None
if network_type == "Custom Adjacency Matrix":
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
    
    adj_matrix_input = st.sidebar.text_area(
        "Adjacency Matrix",
        height=150,
        help="Enter the adjacency matrix as comma-separated values, each row on a new line"
    )
    
    # Process the input adjacency matrix
    if adj_matrix_input:
        try:
            # Parse the input text into a numpy array
            rows = adj_matrix_input.strip().split('\n')
            adj_matrix = []
            for row in rows:
                values = [float(val.strip()) for val in row.split(',')]
                adj_matrix.append(values)
            
            adj_matrix = np.array(adj_matrix)
            
            # Validate the adjacency matrix
            if adj_matrix.shape[0] != adj_matrix.shape[1]:
                st.sidebar.error("The adjacency matrix must be square.")
            elif adj_matrix.shape[0] != n_oscillators:
                st.sidebar.error(f"Matrix dimensions ({adj_matrix.shape[0]}x{adj_matrix.shape[1]}) don't match oscillator count ({n_oscillators}).")
            else:
                st.sidebar.success("Adjacency matrix validated successfully!")
        except Exception as e:
            st.sidebar.error(f"Error parsing matrix: {str(e)}")
            adj_matrix = None

# Create tabs for different visualizations
tab1, tab2, tab3 = st.tabs(["Simulation", "About", "Database"])

# Function to simulate model
@st.cache_data(ttl=300)
def run_simulation(n_oscillators, coupling_strength, frequencies, simulation_time, time_step, random_seed, adjacency_matrix=None):
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed,
        adjacency_matrix=adjacency_matrix
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
    random_seed=random_seed,
    adjacency_matrix=adj_matrix
)

# Tab 1: Simulation
with tab1:
    st.markdown("<h2 class='gradient_text2'>Kuramoto Simulation</h2>", unsafe_allow_html=True)
    
    # Create two columns for the two distributions (histograms at the top)
    st.markdown("<h3 class='gradient_text1'>Initial Distributions</h3>", unsafe_allow_html=True)
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.markdown("<h4>Natural Frequency Distribution</h4>", unsafe_allow_html=True)
        
        # Create frequency distribution histogram
        fig_freq, ax_freq = plt.subplots(figsize=(4, 3))
        
        # Use a gradient colormap for the histogram
        n_bins = 15
        counts, bin_edges = np.histogram(frequencies, bins=n_bins)
        
        # Create custom colormap that matches our gradient theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                 ["#00ffee", "#27aaff"], 
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
        <div class='section'>
            <p><b>Mean:</b> {np.mean(frequencies):.4f}</p>
            <p><b>Standard Deviation:</b> {np.std(frequencies):.4f}</p>
            <p><b>Min:</b> {np.min(frequencies):.4f}</p>
            <p><b>Max:</b> {np.max(frequencies):.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with dist_col2:
        st.markdown("<h4>Initial Phase Distribution</h4>", unsafe_allow_html=True)
        
        # Create initial phase distribution histogram
        fig_init_phase, ax_init_phase = plt.subplots(figsize=(4, 3))
        
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
        
        # Add mean phase marker if order parameter is significant
        if initial_r > 0.3:
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
        <div class='section'>
            <p><b>Initial Order Parameter:</b> {initial_r:.4f}</p>
            <p><b>Initial Mean Phase:</b> {initial_psi:.4f}</p>
            <p>The initial phase distribution affects how quickly the system synchronizes.</p>
            <p>A higher initial order parameter generally leads to faster synchronization.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Create oscillator visualization below the histograms
    st.markdown("<h3 class='gradient_text1'>Interactive Visualization</h3>", unsafe_allow_html=True)
    
    # Add a play button for animation
    col1, col2 = st.columns([3, 1])
    
    with col2:
        st.subheader("Animation Controls")
        animation_speed = st.slider(
            "Animation Speed",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            help="Control how fast the animation plays"
        )
        animate = st.button("Play Animation", use_container_width=True)
        stop_animation = st.button("Stop", use_container_width=True)
    
    with col1:
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
        
        st.markdown(f"""
        <div style='background-color: rgba(0,0,0,0.3); padding: 10px; border-radius: 5px;'>
            <span style='font-size: 1.2em;'><b>Time:</b> {current_time:.2f} &nbsp;|&nbsp; <b>Order Parameter:</b> {current_r:.3f}</span>
        </div>
        """, unsafe_allow_html=True)
    
    # Import needed module
    from matplotlib.collections import LineCollection
    
    # Function to create the phase visualization
    def create_phase_plot(time_idx):
        # Create visualization with enhanced visuals for dark theme
        fig_circle = plt.figure(figsize=(10, 5))
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
        phases_at_time = phases[:, time_idx]
        x = np.cos(phases_at_time)
        y = np.sin(phases_at_time)
        
        # Create custom colormap that matches our gradient_text1 theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                     ["#14a5ff", "#8138ff"], 
                                                     N=256)
        
        # Sort oscillators by their natural frequency
        sorted_indices = np.argsort(frequencies)
        # Create colors based on frequency ordering
        color_indices = np.linspace(0, 1, n_oscillators)
        colors = [custom_cmap(idx) for idx in color_indices]
        
        # Map colors to oscillators by frequency order
        oscillator_colors = np.zeros((n_oscillators, 4))  # RGBA colors
        for i, idx in enumerate(sorted_indices):
            oscillator_colors[idx] = colors[i]
        
        # Scatter plot with oscillator colors
        sc = ax_circle.scatter(x, y, facecolors=oscillator_colors, edgecolor='white', s=180, 
                              alpha=0.9, linewidth=1.5, zorder=10)
        
        # Calculate and show order parameter
        r = order_parameter[time_idx]
        psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
        
        # Draw arrow showing mean field with glow effect - using orange color
        # First add glow/shadow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.07, head_length=0.12, fc='#ffaa33', ec='#ffaa33', 
                       width=0.03, alpha=0.3, zorder=5)
        
        # Then add main arrow
        ax_circle.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                       head_width=0.05, head_length=0.1, fc='#ff9500', ec='#ff9500', 
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
        ax_circle.set_title(f'Oscillators at t={times[time_idx]:.2f}', 
                           color='white', fontsize=14, pad=15)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
        
        return fig_circle
    
    # Function to create oscillator phases over time plot (as dots)
    def create_oscillator_phases_plot(time_idx):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Add background
        ax.set_facecolor('#1a1a1a')
        
        # Create transparent bands at phase regions
        for y in [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi]:
            ax.axhspan(y-0.1, y+0.1, color='#222233', alpha=0.4, zorder=0)
        
        # Create custom colormap that matches our gradient_text1 theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                     ["#14a5ff", "#8138ff"], 
                                                     N=256)
        
        # Sort oscillators by their natural frequency for consistent coloring
        sorted_indices = np.argsort(frequencies)
        color_indices = np.linspace(0, 1, n_oscillators)
        oscillator_colors = np.zeros(n_oscillators, dtype=object)
        
        # Assign colors based on frequency order
        for i, idx in enumerate(sorted_indices):
            oscillator_colors[idx] = custom_cmap(color_indices[i])
        
        # Plot all oscillators as dots up to the current time point
        for i in range(n_oscillators):
            color = oscillator_colors[i]
            
            # Plot oscillator phases as filled dots with color gradient
            ax.scatter(times[:time_idx+1], phases[i, :time_idx+1] % (2 * np.pi), 
                      facecolors=color, edgecolor='white', alpha=0.7, s=50, 
                      linewidth=0.5, zorder=5)
            
            # Add a subtle connecting line with low opacity
            ax.plot(times[:time_idx+1], phases[i, :time_idx+1] % (2 * np.pi), 
                   color=color, alpha=0.2, linewidth=0.8, zorder=2)
            
            # Highlight current position with a larger filled marker
            ax.scatter([times[time_idx]], [phases[i, time_idx] % (2 * np.pi)], 
                      s=140, facecolors=color, edgecolor='white', 
                      linewidth=1.5, zorder=15)
        
        # Add labels for key phase positions
        phase_labels = [(0, '0'), (np.pi/2, 'π/2'), (np.pi, 'π'), (3*np.pi/2, '3π/2'), (2*np.pi, '2π')]
        for y, label in phase_labels:
            ax.annotate(label, xy=(-0.02, y), xycoords=('axes fraction', 'data'),
                       fontsize=11, color='white', ha='center', va='center')
        
        # Plot styling
        ax.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel('Phase (mod 2π)', fontsize=13, fontweight='bold', color='white')
        ax.set_title(f'Oscillator Phases at t={times[time_idx]:.2f}', 
                    fontsize=14, fontweight='bold', color='white', pad=15)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.set_xlim(times.min(), times.max())
        
        # Custom grid
        ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
        
        # Add box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
            
        return fig
    
    # Create a function to create order parameter plot over time (as a dot plot)
    def create_order_parameter_plot(time_idx):
        fig, ax = plt.subplots(figsize=(10, 5))
        
        # Add background gradient
        ax.set_facecolor('#1a1a1a')
        
        # Add subtle horizontal bands for visual reference
        for y in np.linspace(0, 1, 6):
            ax.axhspan(y-0.05, y+0.05, color='#222233', alpha=0.3, zorder=0)
        
        # Create a custom colormap that matches gradient_text2
        cmap = LinearSegmentedColormap.from_list("order_param", 
                                              ["#00ffee", "#27aaff"], 
                                              N=256)
        
        # Plot order parameter with filled gradient dots
        colors = [cmap(r) for r in order_parameter[:time_idx+1]]
        scatter = ax.scatter(times[:time_idx+1], order_parameter[:time_idx+1],
                           facecolors=colors, edgecolor='white',
                           s=70, alpha=0.9, linewidth=0.5, zorder=10)
        
        # Add a connecting line with low opacity
        ax.plot(times[:time_idx+1], order_parameter[:time_idx+1], 
               color='white', alpha=0.3, linewidth=1, zorder=5)
        
        # Highlight current position with a larger filled marker
        if time_idx > 0:
            ax.scatter([times[time_idx]], [order_parameter[time_idx]], 
                     s=180, facecolors=cmap(order_parameter[time_idx]), 
                     edgecolor='white', 
                     linewidth=1.5, zorder=15)
        
        # Add highlights at important thresholds
        ax.axhline(y=0.5, color='#aaaaaa', linestyle='--', alpha=0.5, zorder=1, 
                  label='Partial Synchronization (r=0.5)')
        ax.axhline(y=0.8, color='#ffffff', linestyle='--', alpha=0.5, zorder=1,
                  label='Strong Synchronization (r=0.8)')
        
        # Enhance the plot appearance
        ax.set_xlim(times.min(), times.max())
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Time', fontsize=13, fontweight='bold', color='white')
        ax.set_ylabel('Order Parameter r(t)', fontsize=13, fontweight='bold', color='white')
        ax.set_title(f'Phase Synchronization at t={times[time_idx]:.2f}', 
                    fontsize=14, fontweight='bold', color='white', pad=15)
        
        # Create custom grid
        ax.grid(True, color='#333333', alpha=0.5, linestyle=':')
        
        # Add legend
        ax.legend(loc='upper right', framealpha=0.7)
        
        # Add subtle box around the plot
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
            spine.set_linewidth(1)
        
        # Close any previous figure to avoid memory issues
        plt.close('all')
            
        return fig
    
    # Create three plots stacked vertically
    circle_plot_placeholder = st.empty()
    phases_plot_placeholder = st.empty()
    order_plot_placeholder = st.empty()
    
    # Display initial plots
    circle_plot_placeholder.pyplot(create_phase_plot(time_index))
    phases_plot_placeholder.pyplot(create_oscillator_phases_plot(time_index))
    order_plot_placeholder.pyplot(create_order_parameter_plot(time_index))
    
    # If animation is triggered
    if animate:
        # Store current position to return to after animation
        current_pos = time_index
        
        # Calculate how many frames to skip based on speed
        frame_skip = max(1, int(11 - animation_speed))
        
        # Set up a progress bar
        progress_bar = st.progress(0)
        
        # Animation loop
        for i in range(0, len(times), frame_skip):
            # Check if animation should stop
            if stop_animation:
                break
                
            # Update progress bar
            progress = i / (len(times) - 1)
            progress_bar.progress(progress)
            
            # Update all three plots
            circle_plot_placeholder.pyplot(create_phase_plot(i))
            phases_plot_placeholder.pyplot(create_oscillator_phases_plot(i))
            order_plot_placeholder.pyplot(create_order_parameter_plot(i))
            
            # Add a short pause to control animation speed
            time.sleep(0.1 / animation_speed)
        
        # Clear progress bar after animation
        progress_bar.empty()
    
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Visualization Guide</h3>
        <p>The <b>top plot</b> shows oscillators on a unit circle. Each colored dot represents an oscillator at its current phase position, with colors based on the oscillator's natural frequency ordering.</p>
        <p>The <b>middle plot</b> shows oscillator phases over time as dots. Each horizontal trace represents one oscillator's phase trajectory with consistent coloring.</p>
        <p>The <b>bottom plot</b> shows the order parameter over time, with color-coded dots showing the synchronization level.</p>
        <p>The orange arrow in the circle plot shows the mean field vector, with length equal to the order parameter r.</p>
        <p>Use the slider to manually explore different time points or click "Play Animation" to watch all three visualizations animate together.</p>
    </div>
    """, unsafe_allow_html=True)
    
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
    
    # Show network structure if custom adjacency matrix is used
    if network_type == "Custom Adjacency Matrix" and adj_matrix is not None:
        st.markdown("<h3 class='gradient_text2'>Network Structure</h3>", unsafe_allow_html=True)
        
        # Create a network visualization
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create a heatmap of the adjacency matrix
        im = ax.imshow(adj_matrix, cmap='viridis')
        plt.colorbar(im, ax=ax, label='Connection Strength')
        
        # Add labels and styling
        ax.set_title('Oscillator Network Connectivity', color='white', fontsize=14)
        ax.set_xlabel('Oscillator Index', color='white')
        ax.set_ylabel('Oscillator Index', color='white')
        
        # Set background color
        ax.set_facecolor('#1a1a1a')
        fig.patch.set_facecolor('#121212')
        
        # Add a grid to help distinguish cells
        ax.grid(False)
        
        # Add text annotations for connection strength
        for i in range(adj_matrix.shape[0]):
            for j in range(adj_matrix.shape[1]):
                if adj_matrix[i, j] > 0:
                    ax.text(j, i, f"{adj_matrix[i, j]:.1f}", 
                            ha="center", va="center", 
                            color="white" if adj_matrix[i, j] < 0.7 else "black",
                            fontsize=9)
        
        st.pyplot(fig)
        
        st.markdown("""
        <div class='section'>
            <p>The heatmap above shows the connection strength between oscillators, where:</p>
            <ul>
                <li>Each cell (i,j) represents the coupling strength from oscillator j to oscillator i</li>
                <li>Brighter colors indicate stronger coupling</li>
                <li>Dark/black cells indicate no coupling</li>
            </ul>
            <p>The structure of this network affects how synchronization patterns emerge.</p>
        </div>
        """, unsafe_allow_html=True)

# Tab 2: About
with tab2:
    st.markdown("<h2 class='gradient_text2'>About the Kuramoto Model</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
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
    
    # Applications section
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Applications of the Kuramoto Model</h3>
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
    
    # References section
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>References</h3>
        <div class='section-content'>
            <ul>
                <li>Kuramoto, Y. (1975). Self-entrainment of a population of coupled non-linear oscillators.</li>
                <li>Strogatz, S. H. (2000). From Kuramoto to Crawford: exploring the onset of synchronization in populations of coupled oscillators.</li>
                <li>Acebrón, J. A., et al. (2005). The Kuramoto model: A simple paradigm for synchronization phenomena.</li>
            </ul>
        </div>
    </div>
    """, unsafe_allow_html=True)
    dist_col1, dist_col2 = st.columns(2)
    
    with dist_col1:
        st.markdown("<h3>Natural Frequency Distribution</h3>", unsafe_allow_html=True)
        
        # Create frequency distribution histogram
        fig_freq, ax_freq = plt.subplots(figsize=(6, 5))
        
        # Use a gradient colormap for the histogram
        n_bins = 15
        counts, bin_edges = np.histogram(frequencies, bins=n_bins)
        
        # Create custom colormap that matches our gradient theme
        custom_cmap = LinearSegmentedColormap.from_list("kuramoto_colors", 
                                                  ["#00ffee", "#27aaff"], 
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
        <div class='section'>
            <p><b>Mean:</b> {np.mean(frequencies):.4f}</p>
            <p><b>Standard Deviation:</b> {np.std(frequencies):.4f}</p>
            <p><b>Min:</b> {np.min(frequencies):.4f}</p>
            <p><b>Max:</b> {np.max(frequencies):.4f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with dist_col2:
        st.markdown("<h3>Initial Phase Distribution</h3>", unsafe_allow_html=True)
        
        # Create initial phase distribution histogram
        fig_init_phase, ax_init_phase = plt.subplots(figsize=(6, 5))
        
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
        
        # Add mean phase marker if order parameter is significant
        if initial_r > 0.3:
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
        <div class='section'>
            <p><b>Initial Order Parameter:</b> {initial_r:.4f}</p>
            <p><b>Initial Mean Phase:</b> {initial_psi:.4f}</p>
            <p>The initial phase distribution affects how quickly the system synchronizes.</p>
            <p>A higher initial order parameter generally leads to faster synchronization.</p>
        </div>
        """, unsafe_allow_html=True)

# Database tab
with tab3:
    st.markdown("<h2 class='gradient_text2'>Database</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Store and Retrieve Simulations</h3>
        <p>Save your simulation parameters and results to the database for future reference and comparison.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create two columns for saving and loading
    db_col1, db_col2 = st.columns(2)
    
    with db_col1:
        st.markdown("<h4>Save Current Simulation</h4>", unsafe_allow_html=True)
        
        # Get parameters of frequency distribution based on selected type
        freq_params = {}
        if freq_type == "Normal":
            freq_params = {"mean": freq_mean, "std": freq_std}
        elif freq_type == "Uniform":
            freq_params = {"min": freq_min, "max": freq_max}
        elif freq_type == "Bimodal":
            freq_params = {"peak1": peak1, "peak2": peak2}
        elif freq_type == "Custom":
            freq_params = {"custom_values": custom_freqs}
        
        # Button to save current simulation
        if st.button("Save to Database", use_container_width=True):
            # Store the simulation in the database
            sim_id = store_simulation(
                model=model, 
                times=times, 
                phases=phases, 
                order_parameter=order_parameter,
                frequencies=frequencies,
                freq_type=freq_type,
                freq_params=freq_params,
                adjacency_matrix=adj_matrix  # Store custom adjacency matrix if provided
            )
            
            if sim_id:
                st.success(f"Simulation saved successfully with ID: {sim_id}")
            else:
                st.error("Failed to save simulation to database.")
    
    with db_col2:
        st.markdown("<h4>Load Saved Simulations</h4>", unsafe_allow_html=True)
        
        # Get list of saved simulations
        simulations = list_simulations()
        
        if not simulations:
            st.info("No saved simulations found in the database.")
        else:
            # Create a selection box for the simulations
            sim_options = [f"ID: {sim['id']} - {sim['n_oscillators']} oscillators, K={sim['coupling_strength']}, {sim['timestamp'].strftime('%Y-%m-%d %H:%M')}" 
                         for sim in simulations]
            
            selected_sim = st.selectbox("Select a simulation to load", sim_options)
            
            if selected_sim:
                # Extract simulation ID from selection
                sim_id = int(selected_sim.split("ID: ")[1].split(" -")[0])
                
                if st.button("Load Simulation", use_container_width=True):
                    # Load the simulation from the database
                    sim_data = get_simulation(sim_id)
                    
                    if sim_data:
                        # Display basic information about the loaded simulation
                        st.markdown(f"""
                        <div class='section'>
                            <h4>Simulation Details</h4>
                            <p><b>ID:</b> {sim_data['id']}</p>
                            <p><b>Timestamp:</b> {sim_data['timestamp']}</p>
                            <p><b>Oscillators:</b> {sim_data['n_oscillators']}</p>
                            <p><b>Coupling Strength:</b> {sim_data['coupling_strength']}</p>
                            <p><b>Frequency Distribution:</b> {sim_data['frequency_distribution']}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create visualization of the loaded simulation
                        st.markdown("<h4>Order Parameter</h4>", unsafe_allow_html=True)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        ax.set_facecolor('#1a1a1a')
                        
                        # Create custom colormap
                        cmap = LinearSegmentedColormap.from_list("order_param", 
                                                            ["#00ffee", "#27aaff"], 
                                                            N=256)
                        
                        # Plot order parameter
                        ax.plot(range(len(sim_data['order_parameter']['r'])), 
                              sim_data['order_parameter']['r'], 
                              color=cmap(0.5), linewidth=3)
                        
                        ax.set_xlabel('Time Index', fontsize=12, fontweight='bold', color='white')
                        ax.set_ylabel('Order Parameter r(t)', fontsize=12, fontweight='bold', color='white')
                        ax.set_title('Phase Synchronization', fontsize=14, fontweight='bold', color='white')
                        ax.grid(True, color='#333333', alpha=0.4, linestyle=':')
                        
                        # Add a subtle box around the plot
                        for spine in ax.spines.values():
                            spine.set_edgecolor('#555555')
                            spine.set_linewidth(1)
                        
                        st.pyplot(fig)
                    else:
                        st.error("Failed to load simulation from database.")
    
    # Add section for database management
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Database Management</h3>
        <p>Manage your saved simulations and keep your database organized.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Display all simulations in a table
    if simulations:
        st.markdown("<h4>Saved Simulations</h4>", unsafe_allow_html=True)
        
        # Create a DataFrame for display
        import pandas as pd
        sim_df = pd.DataFrame([
            {
                "ID": sim["id"],
                "Date": sim["timestamp"].strftime("%Y-%m-%d"),
                "Time": sim["timestamp"].strftime("%H:%M:%S"),
                "Oscillators": sim["n_oscillators"],
                "Coupling": f"{sim['coupling_strength']:.2f}",
                "Distribution": sim["frequency_distribution"]
            } for sim in simulations
        ])
        
        st.dataframe(sim_df, use_container_width=True)
        
        # Delete simulation functionality
        st.markdown("<h4>Delete Simulation</h4>", unsafe_allow_html=True)
        
        delete_id = st.number_input("Enter Simulation ID to Delete", min_value=1, step=1)
        
        if st.button("Delete", use_container_width=True):
            # Confirm before deletion
            if delete_id in [sim["id"] for sim in simulations]:
                success = delete_simulation(delete_id)
                if success:
                    st.success(f"Simulation with ID {delete_id} was deleted successfully.")
                    st.rerun()  # Refresh the page to update the list
                else:
                    st.error("Failed to delete simulation.")
            else:
                st.warning(f"No simulation found with ID {delete_id}.")

# Add footer
st.markdown("""
<div style='text-align: center; margin-top: 30px; padding: 10px; font-size: 0.8em;'>
    <p>Kuramoto Model Simulator © 2023 | Interactive Application</p>
</div>
""", unsafe_allow_html=True)
