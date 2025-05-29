import streamlit as st

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="Kuramoto Model Simulator",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Import our organized modules
from src.components.sidebar import render_sidebar
from src.components.animation_tab import render_animation_tab
from src.utils.simulation import run_simulation
from src.utils.plotting import create_network_plot, create_frequency_distribution_plot

# Load CSS
try:
    with open('src/styles/styles.css', 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
except:
    pass

# Main app
def main():
    st.title("🔄 Kuramoto Model Simulator")
    st.markdown("### Interactive exploration of coupled oscillator dynamics")
    
    # Render sidebar and get parameters
    params = render_sidebar()
    
    # Store current params in session state for configuration saving
    st.session_state.current_params = params
    
    # Run simulation
    try:
        model, times, phases, order_parameter = run_simulation(
            n_oscillators=params['n_oscillators'],
            coupling_strength=params['coupling_strength'],
            frequencies=params['frequencies'],
            simulation_time=params['simulation_time'],
            random_seed=params['random_seed'],
            adjacency_matrix=params['adjacency_matrix']
        )
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Network", "Distributions", "Animation", "Numerical"])
        
        with tab1:
            st.subheader("Network Structure")
            network_fig = create_network_plot(params['adjacency_matrix'], params['frequencies'])
            st.pyplot(network_fig)
            
            # Network statistics
            n_nodes = params['adjacency_matrix'].shape[0]
            n_edges = int(np.sum(params['adjacency_matrix']) / 2)
            density = n_edges / (n_nodes * (n_nodes - 1) / 2) if n_nodes > 1 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Nodes", n_nodes)
            col2.metric("Edges", n_edges)
            col3.metric("Density", f"{density:.3f}")
        
        with tab2:
            st.subheader("Frequency Distributions")
            freq_fig = create_frequency_distribution_plot(
                params['frequencies'], 
                params['frequency_distribution'],
                params['freq_params']
            )
            st.pyplot(freq_fig)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Mean", f"{np.mean(params['frequencies']):.3f}")
            col2.metric("Std", f"{np.std(params['frequencies']):.3f}")
            col3.metric("Min", f"{np.min(params['frequencies']):.3f}")
            col4.metric("Max", f"{np.max(params['frequencies']):.3f}")
        
        with tab3:
            st.subheader("Phase Dynamics Animation")
            render_animation_tab(model, times, phases, order_parameter, 
                               params['frequencies'], params['adjacency_matrix'])
        
        with tab4:
            st.subheader("Numerical Considerations")
            st.markdown("""
            **Automatic Time Step Calculation:**
            - Time step is automatically calculated based on oscillator frequencies
            - Ensures numerical stability and accuracy
            - No manual adjustment needed
            
            **Integration Method:**
            - Uses adaptive Runge-Kutta (RK45) method
            - Automatic error control
            - Optimized for performance
            """)
            
            # Show simulation stats
            freq_max = np.max(np.abs(params['frequencies']))
            lambda_max = freq_max + params['coupling_strength']
            time_step = 0.5 / lambda_max if lambda_max > 0 else 0.01
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Max Frequency", f"{freq_max:.3f}")
            col2.metric("Estimated λ_max", f"{lambda_max:.3f}")
            col3.metric("Effective Time Step", f"{time_step:.5f}")
    
    except Exception as e:
        st.error(f"Simulation failed: {str(e)}")
        st.info("Please check your parameters and try again.")

if __name__ == "__main__":
    main()