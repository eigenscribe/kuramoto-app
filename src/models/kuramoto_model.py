import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class KuramotoModel:
    """
    Class to simulate the Kuramoto model of coupled oscillators.
    The Kuramoto model describes the phase dynamics of a system of coupled oscillators.
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
            np.random.seed(random_seed)
            
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
