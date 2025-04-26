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

        # natural frequencies ω_i
        if frequencies is None:
            self.ω = np.random.normal(0, 1, self.N)
        else:
            self.ω = np.array(frequencies, dtype=float)

        # adjacency matrix A (all‐to‐all by default)
        if adjacency_matrix is None:
            A = np.ones((self.N, self.N))
            np.fill_diagonal(A, 0)
            self.A = A
        else:
            self.A = np.array(adjacency_matrix, dtype=float)
            if self.A.shape != (self.N, self.N):
                raise ValueError("Adjacency matrix must be NxN")

        # initial random phases θ(0)
        self.θ0 = 2*np.pi*np.random.rand(self.N)

        # placeholders
        self.times = None
        self.phases = None
        self.order = None

    def kuramoto_rhs(self, t, θ):
        """
        Vectorized Kuramoto RHS: dθ_i/dt = ω_i + (K/N) * sum_j A_ij * sin(θ_j - θ_i)
        """
        Δ = θ[:, None] - θ[None, :]          # shape (N,N): θ[i] - θ[j]
        coupling = (self.K/self.N) * self.A * np.sin(-Δ)
        dθ = self.ω + np.sum(coupling, axis=1)
        return dθ
        
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
        ω_max = np.max(np.abs(self.ω))
        # maximum node–degree in the adjacency
        max_deg = np.max(np.sum(self.A > 0, axis=1))
        λ_max = ω_max + self.K * max_deg

        # 2) Convert that rate to an effective period (avoid div 0)
        if λ_max <= 0:
            T_eff = np.inf
        else:
            T_eff = 2 * np.pi / λ_max

        # 3) Compute our cap: "steps_per_period" per fastest period
        cap = T_eff / steps_per_period

        # 4) If user didn't supply or supplied a larger cap, use ours
        if max_step is None or max_step > cap:
            max_step = cap
            
        # Create a fixed t_eval array to ensure we have enough points for visualization
        t_eval = np.linspace(0, self.T, min_time_points)
        
        print(f"Simulating with max_step={max_step:.5f}, ω_max={ω_max:.5f}, λ_max={λ_max:.5f}")
        print(f"Using t_eval with {len(t_eval)} points")

        # 5) Call the integrator with this max_step and t_eval for fixed time points
        sol = solve_ivp(
            fun=self._rhs,
            t_span=(0, self.T),
            y0=self.θ0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            max_step=max_step,
            t_eval=t_eval
        )

        self.times = sol.t
        self.phases = sol.y
        # order parameter r(t) = |∑ e^{iθ_j}| / N
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
        ax.set_ylabel("Phase (mod 2π)")
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
    t, θ, r = model.simulate()
    model.plot_order_parameter()
    plt.show()