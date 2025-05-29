"""
Animation tab component for the Kuramoto Model Simulator.
Contains animation controls and visualization functionality.
"""

import streamlit as st
import numpy as np
import time
from src.utils.plotting import create_phase_plot, create_oscillator_phases_plot, create_order_parameter_plot


def render_animation_tab(model, times, phases, order_parameter, frequencies, adjacency_matrix):
    """Render the animation tab with controls and visualizations."""
    
    # Initialize session state for animation
    if 'time_index' not in st.session_state:
        st.session_state.time_index = 0
    
    if 'refresh_network' not in st.session_state:
        st.session_state.refresh_network = False
    
    # Animation state
    animate = False
    
    # Define plotting functions that use the simulation data
    def create_phase_plot_wrapper(time_idx):
        return create_phase_plot(time_idx, times, phases, order_parameter, frequencies, adjacency_matrix)
    
    def create_oscillator_phases_plot_wrapper(time_idx):
        return create_oscillator_phases_plot(time_idx, times, phases, frequencies)
    
    def create_order_parameter_plot_wrapper(time_idx):
        return create_order_parameter_plot(time_idx, times, order_parameter)
    
    # Create layout for plots
    phases_plot_placeholder = st.empty()
    phases_plot_placeholder.pyplot(create_oscillator_phases_plot_wrapper(st.session_state.time_index))
    
    # Second row: Circle plot and order parameter side by side
    col1, col2 = st.columns(2)
    with col1:
        circle_plot_placeholder = st.empty()
        circle_plot_placeholder.pyplot(create_phase_plot_wrapper(st.session_state.time_index))
    
    with col2:
        order_plot_placeholder = st.empty()
        order_plot_placeholder.pyplot(create_order_parameter_plot_wrapper(st.session_state.time_index))
    
    # Animation controls
    st.markdown("<h4 style='margin-bottom: 20px;'>Animation Controls</h4>", unsafe_allow_html=True)
    
    button_container = st.container()
    
    with button_container:
        bcol1, bcol2, bcol3, bcol4, bcol5 = st.columns([1, 3, 3, 3, 1])
    
        animation_speed = 5.0  # Faster speed for smoother animation
        
        # Previous frame button
        if bcol2.button("⏪ Previous", use_container_width=True):
            if st.session_state.time_index > 0:
                st.session_state.time_index -= 1
                st.rerun()
        
        # Play button
        play_button_text = "⏯️ Play"
        if bcol3.button(play_button_text, use_container_width=True):
            animate = True
        
        # Next frame button 
        if bcol4.button("⏩ Next", use_container_width=True):
            if st.session_state.time_index < len(times) - 1:
                st.session_state.time_index += 1
                st.rerun()
    
    # Time step display
    time_info_container = st.container()
    with time_info_container:
        time_col1, time_col2, time_col3 = st.columns([1, 2, 1])
        current_time_placeholder = time_col2.empty()
        
        def update_time_step_display(time_idx):
            if time_idx > 0:
                time_step = times[time_idx] - times[time_idx-1]
            else:
                if len(times) > 1:
                    time_step = times[1] - times[0]
                else:
                    time_step = 0
                    
            current_time = times[time_idx]
            
            current_time_placeholder.markdown(f"""
            <div style="padding: 10px; border-radius: 5px; background: linear-gradient(135deg, rgba(138, 43, 226, 0.2), rgba(255, 0, 255, 0.2)); 
                        border: 1px solid rgba(255, 255, 255, 0.1); text-align: center;">
                <span style="font-size: 0.85rem; color: white;">Δt = {time_step:.5f}</span><br>
                <span style="font-size: 0.75rem; color: rgba(255, 255, 255, 0.7);">t = {current_time:.3f}</span>
            </div>
            """, unsafe_allow_html=True)
    
    # Initial display of time step
    update_time_step_display(st.session_state.time_index)
    
    # Time slider
    playback_container = st.container()
    time_index = playback_container.slider(
        "Time Point", 
        min_value=0, 
        max_value=len(times)-1, 
        value=st.session_state.time_index,
        help="Manually select a specific time point to display"
    )
    
    # Update session state when slider is moved
    if st.session_state.time_index != time_index:
        st.session_state.time_index = time_index
    
    # Animation logic
    if animate:
        start_idx = st.session_state.time_index
        
        max_valid_index = len(times) - 1
        if start_idx > max_valid_index:
            start_idx = 0
            st.session_state.time_index = 0
        
        # More aggressive frame skipping for smoothness
        frame_skip = max(1, len(times) // 100)
        
        progress_bar = st.progress(0)
        
        for plot_idx in range(start_idx, len(times), frame_skip):
            st.session_state.time_index = plot_idx
            
            # Update all plots
            phases_plot_placeholder.pyplot(create_oscillator_phases_plot_wrapper(plot_idx))
            circle_plot_placeholder.pyplot(create_phase_plot_wrapper(plot_idx))
            order_plot_placeholder.pyplot(create_order_parameter_plot_wrapper(plot_idx))
            
            # Update progress
            progress = (plot_idx - start_idx) / (len(times) - start_idx)
            progress_bar.progress(progress)
            
            update_time_step_display(plot_idx)
            
            # Much faster frame updates
            time.sleep(0.02 / animation_speed)
        
        progress_bar.empty()
    
    # Visualization guide
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Visualization Guide</h3>
        <p>The <b>top plot</b> shows oscillator phases over time. Each horizontal trace represents one oscillator's phase trajectory with consistent coloring based on the oscillator's natural frequency.</p>
        <p>The <b>bottom left plot</b> shows oscillators on a unit circle. Each colored dot represents an oscillator at its current phase position. The blue arrow shows the mean field vector, with length equal to the order parameter r.</p>
        <p>The <b>bottom right plot</b> shows the order parameter over time, with color-coded dots showing the synchronization level from 0 (no synchronization) to 1 (complete synchronization).</p>
        <p>Click "⏯️ Play" to watch all three visualizations animate together to see the synchronization process in real-time.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Order parameter analysis
    st.markdown("""
    <div class='section'>
        <h3 class='gradient_text1'>Order Parameter Analysis</h3>
        <p>The order parameter r(t) measures the degree of synchronization among oscillators:</p>
        <ul>
            <li>r = 1: Complete synchronization (all oscillators have the same phase)</li>
            <li>r = 0: Complete desynchronization (phases are uniformly distributed)</li>
            <li>0 < r < 1: Partial synchronization (some clustering of phases)</li>
        </ul>
        <p>Watch how the order parameter evolves as oscillators interact through the network!</p>
    </div>
    """, unsafe_allow_html=True)