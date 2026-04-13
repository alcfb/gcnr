#!/usr/bin/env python3
import gcnr

atm = 1.01325e+5 # Pa
t = 20000 # K

print ('(Ievlev, 1980)')
for p in [100,105,110,120,130]:
    d, = gcnr.eos.U.query (p*atm, t, method='ievlev')
    print (f"U: T={t} K   P={p} bar => D={d:6.3f} kg/m3")

print ('(Parks, 1968)')
for p in [100,105,110,120,130]:
    d, e, cp, cv = gcnr.eos.U.query (p*atm, t, method='parks')
    print (f"U: T={t} K   P={p} bar => D={d:6.3f} kg/m3  E={e:9.3e} J/kg  Cv={cv:9.3e} J/kg/K")

