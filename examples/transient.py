#!/usr/bin/env python3
from gcnr import models

ntp = models.ntp_gcr()

ntp.rho = lambda t: 100.0 if t > 1. else 0.0

ntp.temp_feedback = -50.

ntp.steady_state()

res = ntp.transient (5.0, verbose=False, rtol=1.E-9)

print (f"""
    # Time      : {res.time[-1]:9.3f} s
    # Power     : {res.power_MW[-1]:9.3e} MW
    # Fuel Temp.: {res.temperature_K[-1]:9.3f} K
    # Steps     : {res.successful_steps:6.0f}
    # Rejections: {res.rejected_steps:6.0f}
""")

res.plot()
