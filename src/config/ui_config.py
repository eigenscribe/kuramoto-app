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
    
    # Add wisp.jpg background as base64 encoded image
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
        background-image: url('data:image/jpeg;base64,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');
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