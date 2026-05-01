#!/usr/bin/env python3
from gcnr import gcr1
import numpy as np
import matplotlib.pyplot as plt

# ------ LYAPUNOV COEFFS ------

ntp = gcr1.model()

rho = np.linspace (-1700, 1000, 100)
eps = np.linspace (0,0.5,100)

x, y = np.meshgrid (rho, eps)

eigvals = []
for u in zip(x.ravel(), y.ravel()):
    res = ntp.lyapunov (u)
    eigvals.append (res.real.max())

eigvals = np.array (eigvals).reshape(x.shape)

# ------ PLOT ------

fig, ax = plt.subplots (figsize=(5,5), dpi=160)

ax.set_xlabel("Reactivity")
ax.set_ylabel("Emissivity")

ax.imshow (eigvals, cmap='bwr', vmin=-0.01, vmax=0.01)

plt.tight_layout()

plt.show()

#plt.savefig('stability.png')

