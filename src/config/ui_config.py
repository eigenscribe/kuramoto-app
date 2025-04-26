"""
UI configuration for the Kuramoto Model Simulator.
"""

import streamlit as st
import matplotlib.pyplot as plt

def configure_page():
    """Configure the page layout and styling."""
    # Set page config - must be the first Streamlit command
    st.set_page_config(
        page_title="Kuramoto Model Simulator",
        page_icon="🔄",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Apply custom CSS
    _apply_custom_css()
    
    # Configure matplotlib style
    _configure_matplotlib_style()

def _apply_custom_css():
    """Apply custom CSS styling to the app."""
    # Import Aclonica font from Google Fonts
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Aclonica&display=swap');
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Aclonica&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    # Read and apply CSS from the file
    with open("src/styles/app.css", "r") as f:
        css = f.read()
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        
    # Apply additional CSS to fix the background
    st.markdown("""
    <style>
    /* Override background settings for main elements */
    .main .block-container {
        background-color: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        padding: 20px;
        backdrop-filter: blur(5px);
    }
    
    /* Make sure the background image is visible */
    [data-testid="stAppViewContainer"] {
        background-image: url('/static/images/wisp.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    
    /* Add dark overlay on top of background */
    [data-testid="stAppViewContainer"]::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.7);
        z-index: -1;
    }
    </style>
    """, unsafe_allow_html=True)

def _configure_matplotlib_style():
    """Configure matplotlib style for consistent plots."""
    # Set up Matplotlib style for dark theme plots
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#121212',
        'figure.facecolor': '#121212',
        'savefig.facecolor': '#121212',
        'axes.grid': True,
        'grid.color': '#444444',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': '#444444',
        'xtick.color': '#ffffff',
        'ytick.color': '#ffffff',
        'text.color': '#ffffff',
        'axes.labelcolor': '#ffffff',
        'axes.titlecolor': '#ffffff',
        'lines.linewidth': 2,
        'axes.prop_cycle': plt.cycler(color=['#00e8ff', '#14b5ff', '#3a98ff', '#0070eb', 
                                            '#00c3ff', '#0099ff', '#007ffc', '#4169e1']),
    })