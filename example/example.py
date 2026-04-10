#!/usr/bin/env python3
import gcnr

p=105
t=18000

for p in [100,105,110,120,130]:
    print (f"T={t} K   P={p} bar => D={gcnr.eos.query(p, t)} g/cm3")