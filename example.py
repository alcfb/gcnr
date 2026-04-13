#!/usr/bin/env python3
import gcnr

atm = 1.01325e+5 # Pa
t = 20000 # K

print ('(Ievlev, 1980)')
for p in [100,105,110,120,130]:
    d1 = gcnr.eos.U.query (p*atm, t, method='ievlev')
    print (f"T={t} K   P={p} bar => D[U]={d1:6.3f} kg/m3")

print ('(Parks, 1968)')
for p in [100,105,110,120,130]:
    d1 = gcnr.eos.U.query (p*atm, t, method='parks')
    print (f"T={t} K   P={p} bar => D[U]={d1:6.3f} kg/m3")

