#!/usr/bin/env python3
import numpy as np
import scipy.constants as const
from . import tools

# =============================================================================
# GAS CORE REACTOR MODEL 1
# =============================================================================
#
# 1. Reactor Power:
#
#    dP/dt = (rho - bet) / l0 * P + lam.C
#
# 2. Precursors:
#
#    dC/dt = bet / l0 * P - lam * C
#
# 3. Reactivity:
#
#    rho = rho_0(t) + D0 * (T1 - T)
#
# 4. Thermal Radiation
#
#    C M dT/dt = P + S * e * sigma * (T0^4 - T^4)
#
# =============================================================================

class units:
    pcm = 1e-5  # per cent mille
    pi = np.pi
    sigma = const.sigma

params = {
    'l0'  : 8.35e-4,  # s
    'D0'  :-0.1,      # Fuel temperature coefficient, pcm/K
    'D1'  :-0.3,      # Reflector temperature coefficient, pcm/K
    'cv'  : 5.0e+5,   # heat_capacity, J/kg/K
    'mf'  : 2.0,      # fuel mass, kg
    'T1'  : 18000.,   # nominal fuel temperature, K
    'T0'  : 1000.,    # coolant temperature, K
    'lf'  : 2.0,      # fuel cloud length, m
    'rf'  : 0.4,      # fuel cloud radius, m
    'ef'  : 0.1,      # fuel emissivity, -
    'lam' : [0.1],    # 1/s
    'bet' : [280.],   # pcm
}

params ['lam'] = [1.24667e-02, 2.82917e-02, 4.25244e-02, 1.33042e-01, 2.92467e-01, 6.66488e-01, 1.63478e+00, 3.55460e+00]
params ['bet'] = [21.46, 44.97, 40.39, 53.85, 80.24, 10.45, 15.08, 2.83]

# --- NTP Model ---
class model (tools.Solver):
    def __init__(self, params=params):

        super().__init__()

        self.__dict__.update (params)

        self.lam = np.array (self.lam)
        self.bet = np.array (self.bet)

        rank = len (self.lam) + 2

        # --- State ---
        self.x = np.ones(rank)

        # --- Reactivity ---
        self.rho = lambda t: 0.0

        # --- Buffers ---
        self.eye = np.eye(rank)
        self.mat = np.zeros((rank, rank))
        self.src = np.zeros(rank)

        self.vf = units.pi * self.rf**2 * self.lf
        self.sf = 2 * units.pi * (self.rf * self.lf + self.rf**2)

        self.alpha = self.cv * self.mf
        self.gamma = self.sf * self.ef * const.sigma

    def steady_state (self):
        P = self.sf * self.ef * units.sigma * (self.T1**4 - self.T0**4)
        self.x[0] = P
        self.x[1:-1] = units.pcm * self.bet / self.l0 / self.lam * P
        self.x[-1] = self.T1

    def dynamics(self, h, t, b, x, e):
        """
        Implicit step: x - h f(t,x) = b
        """

        # Reactivity
        rho = self.rho(t) + self.D0 * (x[-1] - self.T1)

        # Matrix A
        A = self.mat

        A.fill(0.0)

        A[0, 0] = (rho - sum(self.bet)) * units.pcm / self.l0
        A[0, 1:-1] = self.lam
        A[1:-1, 0] = self.bet / self.l0 * units.pcm

        np.fill_diagonal (A [1:-1, 1:-1], -self.lam)

        A[-1, 0] = 1.0 / self.alpha
        A[-1,-1] = - self.gamma / self.alpha * x[-1]**3

        # RHS
        rhs = self.src
        rhs[:] = b
        rhs[-1] += h * self.gamma / self.alpha * self.T0**4

        # Solve
        A[:] = self.eye - h * A

        x[:] = np.linalg.solve(A, rhs)

