import numpy as np
from scipy.integrate import solve_ivp

class KuramotoModel:
    def __init__(self,
                 n_oscillators=100,
                 coupling_strength=1.0,
                 frequencies=None,
                 adjacency_matrix=None,
                 simulation_time=10.0):
        self.N = n_oscillators
        self.K = coupling_strength
        self.T = simulation_time

        # natural frequencies
        self.ω = (frequencies
                  if frequencies is not None
                  else np.random.normal(0, 1, self.N))

        # adjacency
        if adjacency_matrix is None:
            A = np.ones((self.N, self.N))
            np.fill_diagonal(A, 0)
            self.A = A
        else:
            self.A = adjacency_matrix

        # initial phases
        self.θ0 = 2*np.pi*np.random.rand(self.N)

        # placeholders for solution
        self.times = None
        self.phases = None
        self.order = None

    def kuramoto_rhs(self, t, θ):
        # Vectorized coupling term
        Δ = θ[:, None] - θ[None, :]       # θ[i] - θ[j] for all i,j
        coupling = (self.K/self.N) * self.A * np.sin(-Δ)
        dθ = self.ω + np.sum(coupling, axis=1)
        return dθ

    def simulate(self, rtol=1e-6, atol=1e-9, max_step=0.01):
        sol = solve_ivp(
            fun=self.kuramoto_rhs,
            t_span=(0, self.T),
            y0=self.θ0,
            method='RK45',
            rtol=rtol,
            atol=atol,
            max_step=max_step
        )
        self.times, self.phases = sol.t, sol.y
        self.order = np.abs(np.sum(np.exp(1j*self.phases), axis=0))/self.N
        return self.times, self.phases, self.order