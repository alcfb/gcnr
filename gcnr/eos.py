import os
import numpy as np
import scipy.constants as const
from scipy.interpolate import LinearNDInterpolator#, RBFInterpolator
from scipy import optimize

atm = 1.01325e+5 # Pa

class EOS:
    def __init__(self, method):
        self.filename = os.path.join (os.path.dirname (__file__), 'data', f'uranium_eos_{method:}.txt')

    def setup (self, p, t, *pars):
        points = np.array ([p, t]).T
        values = np.array (pars).T
        self.interp = LinearNDInterpolator (points, values)
        #self.interp = RBFInterpolator(self.points, self.values, degree='cubic')

class U_Ievlev(EOS):
    def __init__ (self):

        EOS.__init__(self, 'ievlev')

        p, t, d = np.loadtxt (self.filename).T

        p *= atm   # Pa

        self.setup(p, t, d)

class U_Parks(EOS):
    def __init__ (self):

        EOS.__init__(self, 'parks')

        p, t, d, e, cv, cp = np.loadtxt (self.filename).T

        p *= 1.01325e+5 # 1 atm     => 1 Pa
        d *= 1.00000e+3 # 1 g/cm3   => 1 kg/m3
        e *= 4.18400e+6 # 1 cal/g   => 1 J/kg
        cv*= 4.18400e+6 # 1 cal/g/K => 1 J/kg/K
        cp*= 4.18400e+6 # 1 cal/g/K => 1 J/kg/K

        self.setup (p, t, d, e, cv, cp)


class EOS_U:
    def __init__(self):

        self.methods = {
            'parks': U_Parks(), 
            'ievlev' : U_Ievlev(),
        }

    def query (self, p, t, method='ievlev'):

        den = self.methods [method].interp (p, t) # kg/m3

        if np.isnan(den).any():
            print(f"Warning: Point ({p}, {t}) is outside the data boundary.")

        return den


class EOS_UF6:
    def __init__ (self):
        pass

    def query (self, p, t):
        "Van der Waals gas"
        R = 8.31 # J / K / mol
        M0 = (233. + 6 * 18.998) * 1.e-3 # kg/mol

        tc = 518. # K
        pc = 64.E+5 # Pa

        a = 27 / 64 * tc**2 / pc
        b = tc / 8 / pc

        fun = lambda x: (p + a * x**2) * (1 - x * b) - x * R * t

        sol = optimize.root (fun, 100)

        den = sol.x[0] * M0

        return den


U = EOS_U()

UF6 = EOS_UF6()



