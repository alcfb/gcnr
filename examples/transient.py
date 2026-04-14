#!/usr/bin/env python3
from gcnr import models

ntp = models.ntp_gcr()

ntp.rho = lambda t: 50.0 if t > 10 else 0.0

ntp.steady_state()

res = ntp.transient (100.0, verbose=False)

print(res.power_MW[-1], res.temperature_K[-1])

