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
    # Let's strip down all CSS in the app to the minimum needed
    st.markdown("""
    <style>
        /* No special styling - let Streamlit's defaults handle alignment */
    </style>
    """, unsafe_allow_html=True)
    
    # Import Aclonica font from Google Fonts
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Aclonica&display=swap');
    
    /* Main styling for the entire application */
    body {
        font-family: 'Aclonica', sans-serif;
        margin: 0;
        padding: 0;
        background-color: #0e1117;
        color: white;
    }
    
    /* Gradient text effects for headings */
    .gradient_text1 {
        font-family: 'Aclonica', sans-serif;
        background: linear-gradient(90deg, #ff00cc, #3333ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 1.7em;
        margin-bottom: 10px;
    }
    
    /* Add custom styling for elements */
    .stButton>button {
        font-family: 'Aclonica', sans-serif;
        width: 100%;
    }
    
    /* Make slider thumbs more visible */
    .stSlider>div>div>div {
        background-color: #ff00cc !important;
    }
    </style>
    """, unsafe_allow_html=True)

def _configure_matplotlib_style():
    """Configure matplotlib style for consistent plots."""
    # Set up Matplotlib style for dark theme plots
    plt.style.use('dark_background')
    plt.rcParams.update({
        'axes.facecolor': '#1e1e1e',
        'figure.facecolor': '#1e1e1e',
        'savefig.facecolor': '#1e1e1e',
        'axes.grid': True,
        'grid.color': '#444444',
        'grid.alpha': 0.3,
        'grid.linestyle': '--',
        'axes.edgecolor': '#444444',
        'xtick.color': '#888888',
        'ytick.color': '#888888',
        'text.color': '#ffffff',
        'axes.labelcolor': '#ffffff',
        'axes.titlecolor': '#ffffff',
        'lines.linewidth': 2,
    })