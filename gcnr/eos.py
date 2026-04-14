#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator


# =========================
# Constants
# =========================
class units:
    atm = 1.01325e+5  # Pa
    bar = 1.e+5       # Pa


# =========================
# Result container
# =========================
@dataclass
class State:
    rho: float
    energy: float | None = None
    cv: float | None = None
    cp: float | None = None


# =========================
# Base EOS model (internal)
# =========================
class _EOSModel:
    def __init__(self, filename):
        self.filename = Path(filename)
        self.interp = None
        self._load()

    def _load(self):
        if not self.filename.exists():
            raise FileNotFoundError(f"{self.filename} not found")

        data = np.loadtxt(self.filename).T
        self._build(data)

    def _build(self, data):
        raise NotImplementedError

    def evaluate(self, p, T):
        out = self.interp(p, T)

        if np.isnan(out).any():
            raise ValueError(f"Point (p={p}, T={T}) outside interpolation domain")

        return np.atleast_1d(out)


# =========================
# Ievlev model
# =========================
class _U_Ievlev(_EOSModel):
    def _build(self, data):
        p, T, rho = data
        p *= units.atm

        points = np.column_stack((p, T))
        values = np.column_stack((rho,))

        self.interp = LinearNDInterpolator(points, values)


# =========================
# Parks model
# =========================
class _U_Parks(_EOSModel):
    def _build(self, data):
        p, T, rho, e, cv, cp = data

        # Unit conversions
        p *= units.atm
        rho *= 1e3
        e *= 4.184e6
        cv *= 4.184e6
        cp *= 4.184e6

        points = np.column_stack((p, T))
        values = np.column_stack((rho, e, cv, cp))

        self.interp = LinearNDInterpolator(points, values)


# =========================
# Public Uranium EOS API
# =========================
class UraniumEOS:
    def __init__(self, method="ievlev", data_dir="data"):
        self.method = method

        filename = os.path.join (os.path.dirname (__file__), 'data', f"uranium_eos_{method}.txt")

        if method == "ievlev":
            self._model = _U_Ievlev(filename)

        elif method == "parks":
            self._model = _U_Parks(filename)

        else:
            raise ValueError(f"Unknown method: {method}")

    # -------- units --------
    def _convert_pressure(self, p, unit):
        if unit == "Pa":
            return p
        elif unit == "bar":
            return np.asarray(p) * units.bar
        elif unit == "atm":
            return np.asarray(p) * units.atm
        else:
            raise ValueError(f"Unknown pressure unit: {unit}")

    # -------- core eval --------
    def _eval(self, p, T):
        return self._model.evaluate(p, T)

    # -------- properties --------
    def rho(self, p, T, p_unit="Pa"):
        p = self._convert_pressure(p, p_unit)
        out = self._eval(p, T)
        return out[..., 0]

    def energy(self, p, T, p_unit="Pa"):
        if self.method != "parks":
            raise NotImplementedError("Energy not available for this EOS")

        p = self._convert_pressure(p, p_unit)
        out = self._eval(p, T)
        return out[..., 1]

    def cv(self, p, T, p_unit="Pa"):
        if self.method != "parks":
            raise NotImplementedError("Cv not available for this EOS")

        p = self._convert_pressure(p, p_unit)
        out = self._eval(p, T)
        return out[..., 2]

    def cp(self, p, T, p_unit="Pa"):
        if self.method != "parks":
            raise NotImplementedError("Cp not available for this EOS")

        p = self._convert_pressure(p, p_unit)
        out = self._eval(p, T)
        return out[..., 3]

    # -------- convenience --------
    def state(self, p, T, p_unit="Pa"):
        p = self._convert_pressure(p, p_unit)
        out = self._eval(p, T)

        if self.method == "ievlev":
            return State(rho=out[..., 0])

        elif self.method == "parks":
            return State(
                rho=out[..., 0],
                energy=out[..., 1],
                cv=out[..., 2],
                cp=out[..., 3],
            )

