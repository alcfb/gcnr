#!/usr/bin/env python3
import gcnr

t = 20000

print("(Ievlev, 1980)")
U = gcnr.eos.UraniumEOS(method="ievlev")

for p in [100,105,110,120,130]:
    rho = U.rho(p=p, T=t, p_unit="atm")
    print(f"U: T={t} K   P={p} atm => D={rho:6.3f} kg/m3")


print("(Parks, 1968)")
U = gcnr.eos.UraniumEOS(method="parks")

for p in [100,105,110,120,130]:
    s = U.state(p=p, T=t, p_unit="atm")

    print(
        f"U: T={t} K   P={p} atm => "
        f"D={s.rho:6.3f} kg/m3  "
        f"E={s.energy:9.3e} J/kg  "
        f"Cv={s.cv:9.3e} J/kg/K"
    )