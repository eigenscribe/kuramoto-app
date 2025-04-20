"""
Helper functions for the Kuramoto simulator.
"""

import streamlit as st
import numpy as np
import base64
import os
import json

def load_css(css_file_path):
    """
    Load and inject CSS into Streamlit app.
    
    Parameters:
    -----------
    css_file_path : str
        Path to the CSS file
    """
    with open(css_file_path, "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

def load_image_as_base64(image_path):
    """
    Load an image file as a base64 string.
    
    Parameters:
    -----------
    image_path : str
        Path to the image file
        
    Returns:
    --------
    str
        Base64 encoded image string
    """
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def set_page_config():
    """
    Set Streamlit page configuration.
    """
    st.set_page_config(
        page_title="Kuramoto Model Simulator",
        page_icon="🌀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def run_simulation(model, params, adjacency_matrix=None, frequencies=None):
    """
    Run a Kuramoto model simulation with the specified parameters.
    
    Parameters:
    -----------
    model : KuramotoModel class
        The Kuramoto model class to instantiate
    params : dict
        Simulation parameters
    adjacency_matrix : ndarray, optional
        Custom adjacency matrix for network connectivity
    frequencies : ndarray, optional
        Custom natural frequencies for oscillators
        
    Returns:
    --------
    model : KuramotoModel
        The instantiated model object
    times : ndarray
        Time points of the simulation
    phases : ndarray
        Phases of oscillators at each time point
    order_parameter : ndarray
        Order parameter values over time
    """
    # Extract parameters
    n_oscillators = params.get('n_oscillators', 10)
    coupling_strength = params.get('coupling_strength', 1.0)
    simulation_time = params.get('simulation_time', 20.0)
    time_step = params.get('time_step', 0.05)
    random_seed = params.get('random_seed', 42)
    
    # Instantiate the model
    kuramoto_model = model(
        n_oscillators=n_oscillators,
        coupling_strength=coupling_strength,
        frequencies=frequencies,
        simulation_time=simulation_time,
        time_step=time_step,
        random_seed=random_seed,
        adjacency_matrix=adjacency_matrix
    )
    
    # Run the simulation
    times, phases, order_parameter = kuramoto_model.simulate()
    
    return kuramoto_model, times, phases, order_parameter

def get_frequency_params(params):
    """
    Extract frequency distribution parameters from the simulation parameters.
    
    Parameters:
    -----------
    params : dict
        Simulation parameters
        
    Returns:
    --------
    dict
        Frequency distribution parameters
    str
        Frequency distribution type
    """
    freq_distribution = params.get('frequency_distribution', 'Normal')
    freq_params = params.get('freq_params', {})
    
    return freq_params, freq_distribution