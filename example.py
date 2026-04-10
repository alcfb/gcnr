#!/usr/bin/env python3
import gcnr

atm = 1.01325e+5 # Pa
t = 18000 # K

for p in [100,105,110,120,130]:
    d1 = gcnr.eos.U.query (p*atm, t) * 1.E-3 # g/cm3
    print (f"T={t} K   P={p} bar => D[U]={d1:6.3f} g/cm3")

