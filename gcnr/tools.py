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

    def step_response (self, rho=100, t_end=10):
        self.rho = lambda t: rho
        return self.transient (t_end=t_end, verbose=False, rtol=1.E-9)

    def _print_step(self, t, x):
        print(f"time={t:12.6e}s x={x}")

    def format_output(self, t, x):
        raise NotImplementedError

    # --- Output formatting ---
    def _print_step(self, t, x):
        print(
            f"time={t:12.6e}s "
            f"N={x[0]*1e-6:12.6e} MW "
            f"Temp={x[-1]:12.6e}K"
        )

    def format_output(self, t, x, info):
        return Result({
            "time": t,
            "power_MW": x[:, 0] * 1e-6,
            "precursor": x[:, 1],
            "temperature_K": x[:, -1],
            "rejected_steps": info.rejected_steps,
            "successful_steps": info.successful_steps,
        })

