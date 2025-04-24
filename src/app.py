import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add project root to path for proper imports
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import internal modules
from src.models.kuramoto_model import KuramotoModel
from src.database.database import (
    store_simulation, get_simulation, list_simulations, delete_simulation,
    save_configuration, list_configurations, get_configuration, delete_configuration,
    export_configuration_to_json, import_configuration_from_json
)
from src.utils.ml_helper import analyze_simulation_data

# Import external packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors
from matplotlib.animation import FuncAnimation
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap
from io import BytesIO
import base64
import json
import time
import os
import networkx as nx
from scipy.optimize import minimize_scalar
from matplotlib.collections import LineCollection
import tempfile
import datetime

# Load CSS from file
def load_css(css_file):
    with open(css_file, "r") as f:
        return f.read()

# Apply CSS styles
try:
    css_content = load_css('src/styles/main.css')
except FileNotFoundError:
    # Fallback for direct development
    try:
        css_content = load_css('styles.css')
    except FileNotFoundError:
        css_content = ""
        st.error("Could not load CSS file")

st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# Function to parse JSON parameters input
def parse_json_parameters(json_string):
    """
    Parse a JSON string containing Kuramoto simulation parameters.
    
    Expected format:
    {
        "n_oscillators": 10,
        "coupling_strength": 1.0,
        "network_type": "All-to-All", 
        "simulation_time": 10.0,
        "time_step": 0.1,
        "auto_optimize": true, (optional)
        "random_seed": 42,
        "frequency_distribution": "Normal",
        "frequency_parameters": {
            "mean": 0.0,
            "std": 0.2
        },
        "adjacency_matrix": [[0,1,1,...], [1,0,1,...], ...] (optional)
    }
    
    Returns:
    dict: Parameters dictionary containing all parsed values, or error message
    """