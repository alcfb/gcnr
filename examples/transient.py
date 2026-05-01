#!/usr/bin/env python3
from gcnr import gcr1

ntp = gcr1.model()

ntp.steady_state()

res = ntp.step_response()

print (f"""
    # Time      : {res.time[-1]:9.3f} s
    # Power     : {res.power_MW[-1]:9.3e} MW
    # Fuel Temp.: {res.temperature_K[-1]:9.3f} K
    # Steps     : {res.successful_steps:6.0f}
    # Rejections: {res.rejected_steps:6.0f}
""")

res.plot()
