"""
Kuramoto Model implementation for coupled oscillator simulations.
"""
import numpy as np
from scipy.integrate import solve_ivp

class KuramotoModel:
    """
    A class representing a Kuramoto model of coupled oscillators.
    
    The Kuramoto model is a mathematical model used to describe the behavior of a large 
    set of coupled oscillators.
    """
    
    def __init__(self, n_oscillators, coupling_strength, frequencies=None, 
                 simulation_time=10.0, time_step=None, random_seed=None, 
                 adjacency_matrix=None):
        """
        Initialize a Kuramoto model.
        
        Parameters:
        -----------
        n_oscillators : int
            Number of oscillators in the system
        coupling_strength : float
            Coupling strength (K) between oscillators
        frequencies : ndarray, optional
            Natural frequencies of oscillators. If None, random frequencies 
            will be generated.
        simulation_time : float, optional
            Total simulation time
        time_step : float, optional  
            Time step for simulation. If None, calculated based on frequencies.
        random_seed : int, optional
            Seed for random number generation
        adjacency_matrix : ndarray, optional
            Adjacency matrix defining network topology, shape (n_oscillators, n_oscillators)
        """
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Store model parameters
        self.n_oscillators = n_oscillators
        self.coupling_strength = coupling_strength
        self.simulation_time = simulation_time
        
        # Initialize natural frequencies
        if frequencies is None:
            self.natural_frequencies = np.random.normal(0, 0.1, n_oscillators)
        else:
            self.natural_frequencies = frequencies
        
        # Calculate appropriate time step if not provided
        if time_step is None:
            max_frequency = max(abs(self.natural_frequencies))
            # Use a time step that samples the highest frequency oscillation well
            self.time_step = min(0.1, 1.0 / (10 * max_frequency)) if max_frequency > 0 else 0.1
        else:
            self.time_step = time_step
            
        # Initialize adjacency matrix
        if adjacency_matrix is None:
            # Default to all-to-all coupling
            self.adjacency_matrix = np.ones((n_oscillators, n_oscillators)) - np.eye(n_oscillators)
        else:
            self.adjacency_matrix = adjacency_matrix
            
        # Initialize phases randomly between 0 and 2π
        self.initial_phases = 2 * np.pi * np.random.random(n_oscillators)
        
        # Initialize containers for simulation results
        self.t = None
        self.phases = None
        self.time_points = None
    
    def _kuramoto_ode(self, t, y):
        """
        The Kuramoto ordinary differential equation system.
        
        Parameters:
        -----------
        t : float
            Time
        y : ndarray
            Current phases of oscillators
            
        Returns:
        --------
        ndarray
            Time derivatives of phases
        """
        derivatives = np.zeros(self.n_oscillators)
        
        # Natural frequency contribution
        derivatives = self.natural_frequencies.copy()
        
        # Coupling contribution
        for i in range(self.n_oscillators):
            for j in range(self.n_oscillators):
                if i != j and self.adjacency_matrix[i, j] > 0:
                    # Coupling strength * sin(phase difference) * adjacency weight
                    derivatives[i] += (self.coupling_strength / self.n_oscillators) * \
                                      self.adjacency_matrix[i, j] * \
                                      np.sin(y[j] - y[i])
        
        return derivatives
    
    def simulate(self, num_points=None):
        """
        Simulate the Kuramoto model.
        
        Parameters:
        -----------
        num_points : int, optional
            Number of time points to return. If None, calculated based on time_step.
            
        Returns:
        --------
        tuple
            (time_points, phases, order_parameter)
        """
        # Calculate number of time points if not provided
        if num_points is None:
            num_points = min(2000, max(500, int(50 * self.simulation_time)))
        
        # Define time points for the solution
        t_eval = np.linspace(0, self.simulation_time, num_points)
        
        # Solve the ODE
        sol = solve_ivp(
            self._kuramoto_ode,
            [0, self.simulation_time],
            self.initial_phases,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6, 
            atol=1e-6
        )
        
        # Store the results
        self.time_points = sol.t
        self.phases = sol.y
        
        # Calculate the order parameter
        order = self.order_parameter()
        
        return sol.t, sol.y, order
    
    def order_parameter(self):
        """
        Calculate the Kuramoto order parameter r(t) for the simulated phases.
        
        The order parameter measures the degree of synchronization in the system.
        r = 0 indicates complete desynchronization, r = 1 indicates complete synchronization.
        
        Returns:
        --------
        ndarray
            Order parameter values for each time point
        """
        if self.phases is None:
            raise ValueError("Simulation must be run before calculating order parameter.")
        
        # Convert phases to complex numbers on the unit circle
        complex_phases = np.exp(1j * self.phases)
        
        # Calculate the average complex phase
        order = np.abs(np.mean(complex_phases, axis=0))
        
        return order