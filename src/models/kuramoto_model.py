import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class KuramotoModel:
    """
    Class to simulate the Kuramoto model of coupled oscillators.
    The Kuramoto model describes the phase dynamics of a system of coupled oscillators.
    
    This implementation includes methods for automatic time step optimization to ensure 
    numerical stability and accuracy while minimizing computational cost.
    """
    def __init__(self, n_oscillators=10, coupling_strength=1.0, frequencies=None, 
                 simulation_time=10.0, time_step=0.01, random_seed=None, adjacency_matrix=None):
        """
        Initialize the Kuramoto model.
        
        Parameters:
        -----------
        n_oscillators : int
            Number of oscillators in the system
        coupling_strength : float
            Strength of coupling between oscillators (K parameter)
        frequencies : array-like or None
            Natural frequencies of the oscillators. If None, random frequencies are generated.
        simulation_time : float
            Total time to simulate (in time units)
        time_step : float
            Time step for simulation
        random_seed : int or None
            Seed for random number generation
        adjacency_matrix : ndarray or None
            Adjacency matrix defining the network structure. If None, a fully connected network is used.
        """
        if random_seed is not None:
            # Convert to int to prevent dtype errors
            np.random.seed(int(random_seed))
            
        self.n_oscillators = n_oscillators
        self.coupling_strength = coupling_strength
        
        # Set natural frequencies of oscillators
        if frequencies is None:
            self.frequencies = np.random.normal(0, 1, n_oscillators)
        else:
            self.frequencies = np.array(frequencies)
            
        # Time parameters
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.t_span = (0, simulation_time)
        self.t_eval = np.arange(0, simulation_time, time_step)
        
        # Initialize phases randomly
        self.initial_phases = 2 * np.pi * np.random.random(n_oscillators)
        
        # Save random seed for future reference
        self.random_seed = random_seed
        
        # Results storage
        self.times = None
        self.phases = None
        self.order_parameter = None
        
        # Set adjacency matrix (network structure)
        if adjacency_matrix is None:
            # Default: fully connected network (all-to-all coupling)
            self.adjacency_matrix = np.ones((n_oscillators, n_oscillators))
            # Remove self-connections (set diagonal to 0)
            np.fill_diagonal(self.adjacency_matrix, 0)
        else:
            # Use provided adjacency matrix
            self.adjacency_matrix = np.array(adjacency_matrix)
            # Ensure it has the correct shape
            if self.adjacency_matrix.shape != (n_oscillators, n_oscillators):
                raise ValueError(f"Adjacency matrix must have shape ({n_oscillators}, {n_oscillators})")

    def kuramoto_ode(self, t, y):
        """
        Defines the Kuramoto differential equations.
        
        Parameters:
        -----------
        t : float
            Current time
        y : array-like
            Current phases of oscillators
        
        Returns:
        --------
        dydt : array-like
            Rate of change of phases
        """
        phases = y
        dydt = np.zeros_like(phases)
        
        # Calculate contributions from all oscillators
        for i in range(self.n_oscillators):
            # Natural frequency term
            dydt[i] = self.frequencies[i]
            
            # Coupling term using adjacency matrix
            for j in range(self.n_oscillators):
                if self.adjacency_matrix[i, j] > 0:  # Only consider connected oscillators
                    dydt[i] += (self.coupling_strength * self.adjacency_matrix[i, j] / self.n_oscillators) * np.sin(phases[j] - phases[i])
                
        return dydt
    
    def simulate(self):
        """
        Run the simulation using SciPy's ODE solver.
        
        Returns:
        --------
        times : array-like
            Time points of the simulation
        phases : array-like
            Phases of oscillators at each time point
        order_parameter : array-like
            Order parameter r(t) at each time point
        """
        # Solve the differential equations
        solution = solve_ivp(
            self.kuramoto_ode,
            self.t_span,
            self.initial_phases,
            method='RK45',
            t_eval=self.t_eval
        )
        
        self.times = solution.t
        self.phases = solution.y
        
        # Calculate the order parameter r(t) = |sum(exp(i*theta_j))|/N
        self.order_parameter = np.abs(np.sum(np.exp(1j * self.phases), axis=0)) / self.n_oscillators
        
        return self.times, self.phases, self.order_parameter
    
    def plot_order_parameter(self, ax=None):
        """
        Plot the order parameter r(t) over time.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Axes on which to plot. If None, creates a new figure.
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.order_parameter is None:
            self.simulate()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        ax.plot(self.times, self.order_parameter)
        ax.set_xlabel('Time')
        ax.set_ylabel('Order Parameter r(t)')
        ax.set_title('Phase Synchronization in Kuramoto Model')
        ax.grid(True)
        
        return ax
    
    def plot_phases(self, ax=None, time_indices=None):
        """
        Plot the phases of all oscillators over time.
        
        Parameters:
        -----------
        ax : matplotlib.axes.Axes or None
            Axes on which to plot. If None, creates a new figure.
        time_indices : list or None
            Indices of time points to plot. If None, plots all time points.
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.phases is None:
            self.simulate()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
            
        if time_indices is None:
            for i in range(self.n_oscillators):
                ax.plot(self.times, self.phases[i, :] % (2 * np.pi))
        else:
            for i in range(self.n_oscillators):
                ax.plot(self.times[time_indices], self.phases[i, time_indices] % (2 * np.pi))
                
        ax.set_xlabel('Time')
        ax.set_ylabel('Phase (mod 2π)')
        ax.set_title('Oscillator Phases Over Time')
        ax.set_ylim(0, 2 * np.pi)
        ax.grid(True)
        
        return ax
    
    def compute_optimal_time_step(self, safety_factor=0.8):
        """
        Compute an optimal time step for accurate and stable simulation.
        
        This method analyzes the model parameters to determine an appropriate time step
        that balances computational efficiency with numerical stability and accuracy.
        
        Parameters:
        -----------
        safety_factor : float, optional
            A factor (0-1) to multiply the theoretical maximum time step by for safety.
            Lower values are more conservative; higher values are more aggressive.
            
        Returns:
        --------
        dict
            Dictionary containing optimal time step and additional analysis:
            - 'optimal_time_step': The recommended time step
            - 'stability_factor': Measure of numerical stability (higher is better)
            - 'accuracy_level': Qualitative assessment of accuracy
            - 'computation_level': Qualitative assessment of computational efficiency
            - 'explanation': Text explanation of the recommendation
        """
        # Calculate the maximum frequency difference in the system
        # This determines the fastest dynamics
        if hasattr(self, 'frequencies') and self.frequencies is not None:
            max_frequency = np.max(np.abs(self.frequencies))
            frequency_range = np.max(self.frequencies) - np.min(self.frequencies)
        else:
            # Default to a conservative estimate if frequencies aren't provided
            max_frequency = 1.0
            frequency_range = 2.0
            
        # Consider network connectivity (average node degree affects dynamics)
        if hasattr(self, 'adjacency_matrix') and self.adjacency_matrix is not None:
            # Average node connectivity (average degree)
            avg_connectivity = np.sum(self.adjacency_matrix > 0) / self.n_oscillators
            # Maximum connectivity any single node has
            max_connectivity = np.max(np.sum(self.adjacency_matrix > 0, axis=1))
        else:
            # Assume fully connected network as worst case
            avg_connectivity = self.n_oscillators - 1
            max_connectivity = self.n_oscillators - 1
            
        # Consider coupling strength (stronger coupling requires smaller steps)
        # Maximum Lyapunov exponent estimate for Kuramoto
        max_lyapunov_estimate = max_frequency + self.coupling_strength * max_connectivity
        
        # Base time step on the fastest dynamics in the system
        # Use the most stringent constraint (either frequency distribution or coupling dynamics)
        theoretical_max_time_step = 1.0 / (10 * max(max_frequency, max_lyapunov_estimate))
        
        # Apply safety factor
        optimal_time_step = safety_factor * theoretical_max_time_step
        
        # Round to a nice value (avoid very odd step sizes)
        power = int(np.floor(np.log10(optimal_time_step)))
        # Round to 1 or 2 significant digits
        optimal_time_step = round(optimal_time_step * 10**(-power)) * 10**power
        
        # Prevent extremely small time steps that would be impractical
        min_allowed_dt = 0.001
        if optimal_time_step < min_allowed_dt:
            optimal_time_step = min_allowed_dt
            
        # Determine qualitative measures for stability and efficiency
        # Note: higher stability_factor means MORE stable
        stability_factor = safety_factor * (1.0 / (max_lyapunov_estimate * optimal_time_step))
        
        # Translate numerical values to qualitative assessments
        if stability_factor > 2.0:
            stability_level = "Excellent"
        elif stability_factor > 1.0:
            stability_level = "Good"
        elif stability_factor > 0.7:
            stability_level = "Adequate" 
        else:
            stability_level = "Marginal"
            
        # Accuracy level (based on frequency range coverage)
        points_per_period = 1.0 / (frequency_range * optimal_time_step)
        if points_per_period > 40:
            accuracy_level = "Excellent"
        elif points_per_period > 20:
            accuracy_level = "Good"
        elif points_per_period > 10:
            accuracy_level = "Adequate"
        else:
            accuracy_level = "Minimal"
            
        # Computational efficiency (subjective - based on total steps)
        total_steps = self.simulation_time / optimal_time_step
        if total_steps < 1000:
            computation_level = "Excellent"
        elif total_steps < 5000:
            computation_level = "Good"
        elif total_steps < 20000:
            computation_level = "Moderate"
        else:
            computation_level = "Intensive"
            
        # Create explanation text
        explanation = (
            f"The recommended time step of {optimal_time_step:.6f} is based on:\n"
            f"• Frequency range: {frequency_range:.4f} (max: {max_frequency:.4f})\n"
            f"• Coupling strength: {self.coupling_strength:.4f}\n"
            f"• Network connectivity: {avg_connectivity:.1f} connections per oscillator\n\n"
            f"This time step yields:\n"
            f"• Approximately {points_per_period:.1f} points per oscillation cycle\n"
            f"• {total_steps:.0f} total simulation steps\n\n"
            f"For higher accuracy, decrease the time step.\n"
            f"For faster computation, increase the time step (with caution)."
        )
            
        # Return comprehensive information
        return {
            'optimal_time_step': optimal_time_step,
            'stability_factor': stability_factor,
            'stability_level': stability_level,
            'accuracy_level': accuracy_level, 
            'computation_level': computation_level,
            'explanation': explanation
        }
    
    def set_optimal_time_step(self, safety_factor=0.8):
        """
        Analyze the model and automatically set the time step to an optimal value.
        
        Parameters:
        -----------
        safety_factor : float, optional
            Factor between 0-1 controlling how conservative the time step is.
            Lower values are more conservative.
            
        Returns:
        --------
        dict
            Dictionary containing the optimization results (see compute_optimal_time_step)
        """
        # Compute optimal time step
        result = self.compute_optimal_time_step(safety_factor)
        
        # Update the model with optimal time step
        self.time_step = result['optimal_time_step']
        self.t_eval = np.arange(0, self.simulation_time, self.time_step)
        
        # Return the results for information
        return result
    
    def visualize_oscillators(self, time_idx=0, ax=None):
        """
        Visualize oscillators on a unit circle at a given time.
        
        Parameters:
        -----------
        time_idx : int
            Index of time point to visualize
        ax : matplotlib.axes.Axes or None
            Axes on which to plot. If None, creates a new figure.
            
        Returns:
        --------
        ax : matplotlib.axes.Axes
            The axes containing the plot
        """
        if self.phases is None:
            self.simulate()
            
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
            
        # Draw unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linestyle='--')
        ax.add_patch(circle)
        
        # Plot oscillators
        phases_at_time = self.phases[:, time_idx]
        x = np.cos(phases_at_time)
        y = np.sin(phases_at_time)
        
        # Color oscillators by their natural frequency
        sc = ax.scatter(x, y, c=self.frequencies, cmap='viridis', s=100, zorder=10)
        plt.colorbar(sc, ax=ax, label='Natural Frequency')
        
        # Calculate and show order parameter
        r = self.order_parameter[time_idx]
        psi = np.angle(np.sum(np.exp(1j * phases_at_time))) 
        
        # Draw arrow showing mean field
        ax.arrow(0, 0, r * np.cos(psi), r * np.sin(psi), 
                 head_width=0.05, head_length=0.1, fc='red', ec='red', 
                 width=0.02, zorder=5)
        
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.grid(True)
        ax.set_title(f'Oscillators at time t={self.times[time_idx]:.2f}, r={r:.3f}')
        
        return ax