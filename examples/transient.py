#!/usr/bin/env python3
import gcnr

ntp = gcnr.models.ntp_gcr()

# Example: step reactivity
#ntp.rho = lambda t: 100.0 if t > 10 else 50.0

res = ntp.transient (100.0, verbose=False)

print(res.power_MW)
print(res.temperature_K)

