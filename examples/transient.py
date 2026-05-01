#!/usr/bin/env python3
from gcnr import gcr1

ntp = gcr1.model()

rho = 0
eps = 0.1

res = ntp.step_response(t_end=10, u=[rho, eps], rho=100)

print (f"""
    # Time      : {res.time[-1]:9.3f} s
    # Power     : {res.power_MW[-1]:9.3e} MW
    # Fuel Temp.: {res.temperature_K[-1]:9.3f} K
    # Steps     : {res.successful_steps:6.0f}
    # Rejections: {res.rejected_steps:6.0f}
""")

res.plot()
