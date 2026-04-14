#!/usr/bin/env python3
import numpy as np
import scipy.constants as const
from ninteg import integrate

class units:
    pcm = 1e-5  # per cent mille


# --- Result container (dict + attribute access) ---
class Result(dict):
    def __getattr__(self, key):
        return self[key]

    def plot(self, fname=None, dpi=140):
        """
        Plot selected variables vs time.

        Usage:
            res.plot("power_MW", "temperature_K")
            res.plot()  # plots all except time
        """
        import matplotlib.pyplot as plt

        t = self["time"]

        fig, ax1 = plt.subplots (figsize=(6,4), dpi=dpi)
        ax2 = ax1.twinx()

        ax1.plot (t, self['power_MW'], label='P', linestyle='-', color='blue')

        ax2.plot (t, self['temperature_K'], label='Tf', linestyle='--', color='red')

        ax1.set_xlabel("Time [s]")
        ax1.set_ylabel("P [MW]")
        ax2.set_ylabel("Tf [K]")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2)

        ax1.grid(True)
        plt.tight_layout()
        plt.show()



# --- Solver ---
class Solver:
    def transient(self, t_end=1.0, h0=1e-9, rtol=1e-6, verbose=False):
        """
        Run simulation and return structured result.
        """

        times = []
        states = []

        for t, x, info in integrate((0, t_end), self.x, self.dynamics, h0=h0, rtol=rtol):
            times.append(t)
            states.append(x.copy())

            if verbose:
                self._print_step(t, x)

        t = np.array(times)
        x = np.array(states)

        return self.format_output(t, x, info)

    def _print_step(self, t, x):
        print(f"time={t:12.6e}s x={x}")

    def format_output(self, t, x):
        raise NotImplementedError


# --- NTP Model ---
class ntp_gcr (Solver):
    def __init__(
        self,
        neutron_lifetime=8.35e-4,     # s
        decay_constant=0.1,           # 1/s
        temp_feedback=-20.0,          # pcm/K
        heat_capacity=5.0e5,          # J/kg/K
        mass=2.0,                     # kg
        fuel_temp_ref=1.8e4,          # K
        sink_temp=1e3,                # K
        length=2.0,                   # m
        radius=0.4,                   # m
        emissivity=0.1,               # -
        delayed_fraction=280.0        # pcm
    ):
        super().__init__()

        # --- State ---
        self.x = np.ones(3)

        # --- Reactivity ---
        self.rho = lambda t: 0.0

        # --- Parameters ---
        self.neutron_lifetime = neutron_lifetime
        self.decay_constant = decay_constant
        self.temp_feedback = temp_feedback
        self.heat_capacity = heat_capacity
        self.mass = mass
        self.fuel_temp_ref = fuel_temp_ref
        self.sink_temp = sink_temp
        self.length = length
        self.radius = radius
        self.emissivity = emissivity
        self.delayed_fraction = delayed_fraction

        # --- Buffers ---
        self.eye = np.eye(3)
        self.mat = np.zeros((3, 3))
        self.src = np.zeros(3)

        # --- Precompute ---
        self._precompute()

    def _precompute(self):
        self.V = const.pi * self.radius**2 * self.length
        self.S = 2 * const.pi * (self.radius * self.length + self.radius**2)

        self.rad_coeff = self.S * self.emissivity * const.sigma / (
            self.heat_capacity * self.mass
        )
    def steady_state(self):
        power = self.S * self.emissivity * const.sigma * (self.fuel_temp_ref**4 - self.sink_temp**4)
        self.x[0] = power
        self.x[1] = self.delayed_fraction * units.pcm / self.neutron_lifetime / self.decay_constant * power
        self.x[2] = self.fuel_temp_ref

    def dynamics(self, h, t, b, x, e):
        """
        Implicit step: x - h f(t,x) = b
        """

        # Reactivity
        rho = self.rho(t) + self.temp_feedback * (x[2] - self.fuel_temp_ref)

        # Matrix A
        A = self.mat
        A.fill(0.0)

        A[0, 0] = (rho - self.delayed_fraction) * units.pcm / self.neutron_lifetime
        A[0, 1] = self.decay_constant
        A[1, 0] = self.delayed_fraction * units.pcm / self.neutron_lifetime
        A[1, 1] = -self.decay_constant

        A[2, 0] = 1.0 / (self.heat_capacity * self.mass)
        A[2, 2] = -self.rad_coeff * x[2]**3

        # RHS
        rhs = self.src
        rhs[:] = b
        rhs[2] += h * self.rad_coeff * self.sink_temp**4

        # Solve
        A[:] = self.eye - h * A

        x[:] = np.linalg.solve(A, rhs)

    # --- Output formatting ---
    def _print_step(self, t, x):
        print(
            f"time={t:12.6e}s "
            f"N={x[0]*1e-6:12.6e} MW "
            f"Temp={x[2]:12.6e}K"
        )

    def format_output(self, t, x, info):
        return Result({
            "time": t,
            "power_MW": x[:, 0] * 1e-6,
            "precursor": x[:, 1],
            "temperature_K": x[:, 2],
            "rejected_steps": info.rejected_steps,
            "successful_steps": info.successful_steps,
        })

