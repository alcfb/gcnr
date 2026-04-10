import os
import numpy as np
from scipy.interpolate import LinearNDInterpolator

class UraniumEOS:
    def __init__ (self):

        filename = os.path.join (os.path.dirname (__file__), 'data', 'uranium_eos_ievlev.txt')

        try:
            p, t, d = np.loadtxt (filename).T
        except FileNotFoundError:
            raise FileNotFoundError(f"EOS data file not found at {filename}")

        #p *= 1.01325e+5 # 1 atm     => 1 Pa
        #d *= 1.00000e+3 # 1 g/cm3   => 1 kg/m3
        #e *= 4.18400e+6 # 1 cal/g   => 1 J/kg
        #cv*= 4.18400e+6 # 1 cal/g/K => 1 J/kg/K
        #cp*= 4.18400e+6 # 1 cal/g/K => 1 J/kg/K

        self.points = np.array ([p, t]).T
        self.values = d
        self.interp = LinearNDInterpolator (self.points, self.values)

    def query (self, p=100, t=1.E+4):

        res = self.interp (p, t)

        if np.isnan(res).any():
            print(f"Warning: Point ({p}, {t}) is outside the data boundary.")

        return res

eos = UraniumEOS()