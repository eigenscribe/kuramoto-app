import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class KuramotoModel:
    """
    Kuramoto model with adaptive RK45 integration and an automatic max_step
    based on the fastest natural frequency.
    """
    def __init__(self,
                n_oscillators=10,
                coupling_strength=1.0,
                frequencies=None,
                adjacency_matrix=None,
                simulation_time=10.0,
                random_seed=None):
        if random_seed is not None:
            np.random.seed(int(random_seed))

        self.N = n_oscillators
        self.K = coupling_strength
        self.T = simulation_time

        # natural frequencies for each oscillator
        if frequencies is None:
            self.natural_frequencies = np.random.normal(0, 1, self.N)
        else:
            self.natural_frequencies = np.array(frequencies, dtype=float)

        # adjacency matrix A (all‐to‐all by default)
        if adjacency_matrix is None:
            A = np.ones((self.N, self.N))
            np.fill_diagonal(A, 0)
            self.A = A
        else:
            self.A = np.array(adjacency_matrix, dtype=float)
            if self.A.shape != (self.N, self.N):
                raise ValueError("Adjacency matrix must be NxN")

        # initial random phases for each oscillator
        self.initial_phases = 2*np.pi*np.random.rand(self.N)

        # placeholders
        self.times = None
        self.phases = None
        self.order = None

    def kuramoto_rhs(self, t, phases):
        """
        Vectorized Kuramoto equation: dphase_i/dt = natural_freq_i + (K/N) * sum_j A_ij * sin(phase_j - phase_i)
        """
        # Calculate phase differences between all pairs of oscillators
        phase_diff = phases[:, None] - phases[None, :]
        
        # Calculate coupling term
        coupling = (self.K/self.N) * self.A * np.sin(-phase_diff)
        
        # Rate of change = natural frequency + coupling influence
        dphase_dt = self.natural_frequencies + np.sum(coupling, axis=1)
        
        return dphase_dt
        
    # Alias for backwards compatibility
    _rhs = kuramoto_rhs

    def simulate(self,
         rtol=1e-6,
         atol=1e-9,
         max_step=None,
         steps_per_period=20,
         min_time_points=500):
        """
        Run an adaptive RK45, capping the step‐size by the fastest combined ω+coupling timescale.
        
        Parameters
        ----------
        rtol : float
            Relative tolerance for the solver
        atol : float
            Absolute tolerance for the solver
        max_step : float, optional
            Maximum time step, will be calculated if not provided
        steps_per_period : int
            Number of steps desired per fastest natural‐frequency period
        min_time_points : int
            Minimum number of time points to generate for visualization
        
        Returns
        --------
        tuple
            (times, phases, order_parameter)
        """
        # 1) Estimate fastest rate: natural frequencies + coupling
        freq_max = np.max(np.abs(self.natural_frequencies))
        # maximum node degree in the adjacency
        max_deg = np.max(np.sum(self.A > 0, axis=1))
        lambda_max = freq_max + self.K * max_deg

        # 2) Convert that rate to an effective period (avoid div 0)
        if lambda_max <= 0:
            T_eff = np.inf
        else:
            T_eff = 2 * np.pi / lambda_max

        # 3) Compute our cap: "steps_per_period" per fastest period
        cap = T_eff / steps_per_period

        # 4) If user didn't supply or supplied a larger cap, use ours
        if max_step is None or max_step > cap:
            max_step = cap
            
        # Scale the number of points based on simulation duration
        # For longer simulations, use more points, but cap at 2000 points to avoid performance issues
        points_per_time_unit = min_time_points / 10.0  # 500 points for a 10-unit simulation is our baseline
        max_points = 2000  # Upper limit to prevent performance issues with very long simulations
        scaled_points = min(max_points, max(min_time_points, int(points_per_time_unit * self.T)))
        
        # Create the t_eval array with scaled number of points
        t_eval = np.linspace(0, self.T, scaled_points)
        
        print(f"Simulating with max_step={max_step:.5f}, freq_max={freq_max:.5f}, lambda_max={lambda_max:.5f}")
        print(f"Using t_eval with {len(t_eval)} points for simulation duration {self.T:.1f}")

        # 5) Call the integrator with this max_step and t_eval for fixed time points
        sol = solve_ivp(
            fun=self._rhs,
            t_span=(0, self.T),
            y0=self.initial_phases,
            method='RK45',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval
        )

        self.times = sol.t
        self.phases = sol.y
        # order parameter r(t) = |sum of e^{i*phase_j}| / N
        self.order = np.abs(np.sum(np.exp(1j*self.phases), axis=0)) / self.N
        return self.times, self.phases, self.order

    def plot_order_parameter(self, ax=None):
        if self.order is None:
            self.simulate()
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.times, self.order)
        ax.set_xlabel("Time")
        ax.set_ylabel("Order Parameter r(t)")
        ax.set_title("Kuramoto Synchronization")
        ax.grid(True)
        return ax

    def plot_phases(self, ax=None):
        if self.phases is None:
            self.simulate()
        if ax is None:
            fig, ax = plt.subplots()
        for i in range(self.N):
            ax.plot(self.times, self.phases[i,:] % (2*np.pi), lw=0.8)
        ax.set_xlabel("Time")
        ax.set_ylabel("Phase (mod 2*pi)")
        ax.set_title("Oscillator Phases")
        ax.set_ylim(0, 2*np.pi)
        ax.grid(True)
        return ax

# Example usage
if __name__ == "__main__":
    model = KuramotoModel(n_oscillators=50,
                         coupling_strength=2.5,
                         simulation_time=20.0,
                         random_seed=123)
    t, theta, r = model.simulate()
    model.plot_order_parameter()
    plt.show()