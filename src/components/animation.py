"""
Animation component for the Kuramoto simulator.
This module provides functions for creating and displaying animations of the model.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
import io
import base64
from IPython.display import HTML

def render_animation_tab(model, times, phases, order_parameter):
    """
    Render the animation tab with interactive oscillator animation.
    
    Parameters:
    -----------
    model : KuramotoModel
        The Kuramoto model object
    times : ndarray
        Time points of the simulation
    phases : ndarray
        Phases of oscillators over time
    order_parameter : ndarray
        Order parameter values over time
    """
    st.markdown("## 🎬 Animation")
    st.markdown("### Visualize Oscillator Synchronization")
    
    # Get number of time points and oscillators
    n_times = len(times)
    n_oscillators = model.n_oscillators
    
    # Create time slider
    time_idx = st.slider(
        "Time Point", 
        min_value=0, 
        max_value=n_times-1, 
        value=0, 
        format="t = %.2fs" % times[0],
        key="time_point_slider"
    )
    
    # Update the time display when the slider changes
    st.markdown(f"**Time:** {times[time_idx]:.2f}s")
    
    # Create tabs for different visualizations
    viz_tabs = st.tabs(["Unit Circle", "Phase Evolution", "Order Parameter"])
    
    with viz_tabs[0]:
        st.markdown("### Oscillator Positions on Unit Circle")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up axis
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Oscillator Phases at t={times[time_idx]:.2f}s', fontsize=14)
        
        # Add unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        
        # Get phases at the current time
        phases_at_time = phases[:, time_idx]
        
        # Calculate positions on the unit circle
        x = np.cos(phases_at_time)
        y = np.sin(phases_at_time)
        
        # Create colormap based on natural frequencies
        colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
        
        # Plot lines from origin to each oscillator
        for i in range(n_oscillators):
            ax.plot([0, x[i]], [0, y[i]], color=colors[i], alpha=0.7, lw=2)
        
        # Plot oscillators
        scatter = ax.scatter(x, y, c=model.frequencies, cmap='viridis', s=100, edgecolor='white', linewidth=1, zorder=10)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Natural Frequency')
        
        # Calculate and show order parameter
        r = order_parameter[time_idx]
        psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
        
        # Draw arrow showing mean field
        arrow = ax.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                         head_width=0.05, head_length=0.1, fc='red', ec='red', 
                         width=0.02, zorder=5)
        
        # Add annotation for order parameter
        ax.text(0.02, 0.02, f'r = {r:.3f}', transform=ax.transAxes, fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
        
        st.pyplot(fig)
        
        # Add metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Order Parameter", f"{r:.3f}")
        with col2:
            st.metric("Mean Phase", f"{np.mean(phases_at_time):.3f}")
        with col3:
            st.metric("Phase Coherence", f"{r:.3f}")
    
    with viz_tabs[1]:
        st.markdown("### Phase Evolution Over Time")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot phases for all oscillators
        for i in range(n_oscillators):
            ax.plot(times, phases[i, :] % (2 * np.pi), color=colors[i], alpha=0.7, lw=1)
        
        # Add vertical line for current time
        ax.axvline(x=times[time_idx], color='red', linestyle='--', alpha=0.7)
        
        # Add labels and grid
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Phase (mod 2π)', fontsize=12)
        ax.set_title('Oscillator Phases Over Time', fontsize=14)
        ax.set_ylim(0, 2 * np.pi)
        ax.set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        ax.set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
        ax.grid(True, alpha=0.3)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, norm=plt.Normalize(0, n_oscillators-1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Oscillator Index')
        
        st.pyplot(fig)
        
        # Add metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Time", f"{times[time_idx]:.2f}s")
        with col2:
            st.metric("Phase Variance", f"{np.var(phases[:, time_idx] % (2 * np.pi)):.3f}")
    
    with viz_tabs[2]:
        st.markdown("### Order Parameter Evolution")
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot order parameter
        ax.plot(times, order_parameter, linewidth=2)
        ax.fill_between(times, 0, order_parameter, alpha=0.3)
        
        # Add vertical line for current time
        ax.axvline(x=times[time_idx], color='red', linestyle='--', alpha=0.7)
        
        # Add horizontal line for current order parameter
        ax.axhline(y=order_parameter[time_idx], color='red', linestyle='--', alpha=0.7)
        
        # Add labels and grid
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Order Parameter (r)', fontsize=12)
        ax.set_title('Kuramoto Model Synchronization', fontsize=14)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        # Add annotation for current value
        ax.annotate(
            f'r = {order_parameter[time_idx]:.3f}',
            xy=(times[time_idx], order_parameter[time_idx]),
            xytext=(times[time_idx] + 0.1 * times[-1], order_parameter[time_idx] + 0.1),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2', color='black'),
            bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7)
        )
        
        st.pyplot(fig)
        
        # Add metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current r", f"{order_parameter[time_idx]:.3f}")
        with col2:
            st.metric("Final r", f"{order_parameter[-1]:.3f}")
        with col3:
            # Calculate rate of change
            if time_idx > 0:
                rate = (order_parameter[time_idx] - order_parameter[time_idx-1]) / (times[time_idx] - times[time_idx-1])
                st.metric("Rate of Change", f"{rate:.3f} /s")
            else:
                st.metric("Rate of Change", "N/A")
    
    # Animation section
    st.markdown("---")
    st.markdown("### 🎞️ Create Animation")
    st.markdown("Generate and play an animation of oscillator synchronization.")
    
    # Create animation settings
    col1, col2 = st.columns(2)
    
    with col1:
        anim_speed = st.slider("Animation Speed", 10, 200, 50, step=10)
        anim_duration = st.slider("Animation Duration (s)", 1.0, 10.0, 5.0, step=0.5)
    
    with col2:
        num_frames = st.slider("Number of Frames", 20, 100, 50, step=5)
        smoothness = st.slider("Smoothness", 1, 10, 3, step=1)
    
    # Animation function
    if st.button("Play Animation", type="primary", key="play_animation_button"):
        with st.spinner("Generating animation..."):
            # Create figure and axis
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Set up axis
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add unit circle
            circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
            ax.add_patch(circle)
            
            # Create colormap based on natural frequencies
            colors = plt.cm.viridis(np.linspace(0, 1, n_oscillators))
            
            # Initialize scatter plot
            scatter = ax.scatter([], [], c=model.frequencies, cmap='viridis', s=100, edgecolor='white', linewidth=1, zorder=10)
            
            # Initialize lines
            lines = [ax.plot([], [], color=colors[i], alpha=0.7, lw=2)[0] for i in range(n_oscillators)]
            
            # Initialize order parameter arrow
            arrow = ax.arrow(0, 0, 0, 0, head_width=0.05, head_length=0.1, fc='red', ec='red', width=0.02, zorder=5)
            arrow_patch = arrow.get_patch_transform()
            
            # Initialize text for order parameter value
            r_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=12, 
                            bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
            
            # Calculate time points for the animation
            indices = np.linspace(0, n_times-1, num_frames, dtype=int)
            
            # Animation update function
            def update(frame):
                idx = indices[frame]
                
                # Get phases at the current time
                phases_at_time = phases[:, idx]
                
                # Calculate positions on the unit circle
                x = np.cos(phases_at_time)
                y = np.sin(phases_at_time)
                
                # Update scatter positions
                scatter.set_offsets(np.column_stack([x, y]))
                
                # Update lines
                for i, line in enumerate(lines):
                    line.set_data([0, x[i]], [0, y[i]])
                
                # Calculate order parameter
                r = order_parameter[idx]
                psi = np.angle(np.sum(np.exp(1j * phases_at_time)))
                
                # Remove old arrow and create new one
                arrow.remove()
                arrow_new = ax.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                                     head_width=0.05, head_length=0.1, fc='red', ec='red', 
                                     width=0.02, zorder=5)
                
                # Update text
                r_text.set_text(f'r = {r:.3f}')
                
                # Update title
                ax.set_title(f'Oscillator Phases at t={times[idx]:.2f}s', fontsize=14)
                
                return [scatter, *lines, arrow_new, r_text]
            
            # Create animation
            interval = anim_duration * 1000 / num_frames  # Convert to milliseconds per frame
            anim = FuncAnimation(fig, update, frames=num_frames, interval=interval, blit=False)
            
            # Convert to HTML5 video
            html = anim.to_html5_video()
            
            # Display animation
            st.markdown("### Animation Result")
            st.components.v1.html(html, height=600)
            
            plt.close(fig)
            
            st.success("Animation completed! Play the video above.")
        
    # Additional information
    with st.expander("Animation Tips"):
        st.markdown("""
        - **Speed**: Controls how fast the animation plays.
        - **Duration**: Total length of the animation in seconds.
        - **Frames**: Number of frames in the animation. More frames = smoother but slower to generate.
        - **Smoothness**: Controls the interpolation between frames.
        
        For best results, use 50-100 frames with a duration of 5-10 seconds.
        """)