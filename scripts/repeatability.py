"""How repeatable is "drop loss against the loopback"?  Re-align the fibers many times on one die.

Each fiber of the array has its own alignment loss, and the loopback uses two other fibers than a drop
channel does. The loopback therefore removes the couplers' spectral shape and the routing, but not the
difference between fibers. This script measures what is left.

    python scripts/repeatability.py --wafer W01 --die D07 --bank 2 --n 500
"""
import argparse

import numpy as np

from pts.dut import DUT, N_RINGS, design_resonances_nm

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wafer", default="W01"); ap.add_argument("--die", default="D07")
    ap.add_argument("--bank", type=int, default=2); ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    d = DUT(rng_seed=1)
    d.load(a.wafer, a.die); d.bank = a.bank
    lam = np.arange(1530.0, 1570.0, 0.002)
    tr = d.transmissions(lam)
    des = design_resonances_nm(a.bank)
    print(f"{a.wafer}/{a.die} bank {a.bank}, {a.n} fiber alignments")
    print("ring  true loss (dB)  measured mean  sd     min    max   share below zero")
    for k in range(1, N_RINGS + 1):
        y = tr[f"drop{k}"]
        near = np.abs(lam - des[k]) <= 3.8
        i = np.flatnonzero(near)[np.argmax(y[near])]
        true = -10 * np.log10(y[i])
        meas = []
        for _ in range(a.n):
            d.realign()
            p_drop = d.power_dbm(k, lam[i], 0.0, True)
            p_ref = d.power_dbm(6, lam[i], 0.0, True)
            meas.append(p_ref - p_drop)
        m = np.array(meas)
        print(f"{k}     {true:5.2f}           {m.mean():5.2f}          {m.std():4.2f}  {m.min():5.2f}  {m.max():5.2f}   {100*np.mean(m<0):.0f} %")
