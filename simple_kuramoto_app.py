import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from kuramoto_model import KuramotoModel

# Page config
st.set_page_config(
    page_title="Simple Kuramoto Model",
    page_icon="🔄",
    layout="wide"
)

# Basic styling
st.markdown("""
<style>
    body {
        background-color: #121212;
        color: white;
    }
    .stApp {
        background-color: #121212;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Simple Kuramoto Model Simulator")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")

n_oscillators = st.sidebar.slider("Number of Oscillators", 3, 50, 10)
coupling_strength = st.sidebar.slider("Coupling Strength", 0.0, 5.0, 1.0, 0.1)
simulation_time = st.sidebar.slider("Simulation Time", 1.0, 20.0, 10.0, 0.5)
random_seed = st.sidebar.number_input("Random Seed", 0, 1000, 42)

# Create frequencies (simple normal distribution)
mean = 0.0
std = 0.2
frequencies = np.random.normal(mean, std, n_oscillators)

# Run button
if st.sidebar.button("Run Simulation"):
    st.info("Running simulation...")
    
    # Create model and run simulation
    model = KuramotoModel(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        random_seed=random_seed
    )
    
    times, phases, order_parameter = model.simulate()
    
    # Display results
    st.success("Simulation completed!")
    
    # Create basic plots in two columns
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Phase Evolution")
        fig1, ax1 = plt.subplots(figsize=(8, 4))
        for i in range(n_oscillators):
            ax1.plot(times, phases[i], alpha=0.7)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Phase")
        ax1.set_title("Oscillator Phases")
        ax1.grid(True)
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Order Parameter")
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.plot(times, order_parameter, 'r-')
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Order Parameter (r)")
        ax2.set_title("Synchronization Level")
        ax2.set_ylim(0, 1.1)
        ax2.grid(True)
        st.pyplot(fig2)
    
    # Display final state as a circle
    st.subheader("Final State")
    final_phases = phases[:, -1]
    
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
    ax3.add_artist(circle)
    
    # Plot oscillators on the circle
    x = np.cos(final_phases)
    y = np.sin(final_phases)
    ax3.scatter(x, y, s=100, c=range(n_oscillators), cmap='viridis')
    
    # Plot the order parameter as an arrow
    order_complex = np.mean(np.exp(1j * final_phases))
    order_mag = np.abs(order_complex)
    order_angle = np.angle(order_complex)
    ax3.arrow(0, 0, order_mag * np.cos(order_angle), order_mag * np.sin(order_angle), 
             head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    ax3.set_xlim(-1.2, 1.2)
    ax3.set_ylim(-1.2, 1.2)
    ax3.set_aspect('equal')
    ax3.grid(True)
    ax3.set_title("Final Oscillator Positions")
    st.pyplot(fig3)
    
else:
    st.info("Adjust parameters and click 'Run Simulation' to start.")