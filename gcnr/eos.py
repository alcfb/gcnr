#!/usr/bin/env python3
import os
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from scipy.interpolate import LinearNDInterpolator, interp1d


# =========================
# Constants
# =========================
class units:
    atm = 1.01325e+5  # Pa
    bar = 1.e+5       # Pa
    cal = 4.184e3     # J

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
# Koroteyev model
# =========================
class _U_Koroteyev(_EOSModel):
    def _build(self, data):

        p, T, k = data

        # Unit conversions
        p *= units.atm # Pa
        k *= units.cal * 1.E+3 / 3600. # W/(m K)

        points = np.column_stack((p, T))
        values = np.column_stack((k,))

        self.interp = LinearNDInterpolator(points, values)

        # to extrapolate for values below 100 atm
        # build 1d interpolators for ranges between 100 atm to 200 atm
        p_100 = 100 * units.atm
        p_200 = 200 * units.atm

        mask_100 = np.isclose(p, p_100)
        mask_200 = np.isclose(p, p_200)

        self._T_100 = T[mask_100]
        self._k_100 = k[mask_100]
        self._T_200 = T[mask_200]
        self._k_200 = k[mask_200]

        self._interp_100 = interp1d(self._T_100, self._k_100, kind='linear')
        self._interp_200 = interp1d(self._T_200, self._k_100, kind='linear')

        self._p_100 = p_100
        self._p_200 = p_200

    def extrapolate(self, p, T):
        """Linear extrapolation in pressure for p < 100 atm using 100 & 200 atm data."""
        k_100 = self._interp_100(T)
        k_200 = self._interp_200(T)

        # Linear extrapolation: k(p) = k_lo + (p - p_lo) * (k_hi - k_lo) / (p_hi - p_lo)
        k = k_100 + (p - self._p_100) * (k_200 - k_100) / (self._p_200 - self._p_100)

        return np.atleast_1d(k)


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

        elif method == "koroteyev":
            self._model = _U_Koroteyev(filename)

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
    def conductivity(self, p, T, p_unit="Pa", extrapolation=False):
        if self.method != "koroteyev":
            raise NotImplementedError("Conductivity not available for this EOS")
        p = self._convert_pressure(p, p_unit)

        p_min = 100 * units.atm
        if np.any(np.asarray(p) < p_min):
            if not extrapolation:
                raise ValueError(
                    f"Pressure {p} Pa is below the interpolation domain (100 atm). "
                    f"Set extrapolation=True to enable linear extrapolation."
                )
            return self._model.extrapolate(p, T)

        out = self._eval(p, T)
        return out[..., 0]

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

