"""Measure a lot: every die of one or more wafers, one bank. Prints the yield at the end.

    python scripts/measure_lot.py --wafers W01 W02 --dies 25 --bank 2
"""
import argparse

from pts import db
from pts.routines import measure_die
from pts.station import Station, StationConfig

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wafers", nargs="+", default=["W01"])
    ap.add_argument("--dies", type=int, default=25)
    ap.add_argument("--bank", type=int, default=2)
    ap.add_argument("--config", default="config/virtual.json")
    ap.add_argument("--db", default="data/results.sqlite")
    ap.add_argument("--no-heater", action="store_true")
    a = ap.parse_args()
    con = db.connect(a.db)
    n_pass = n = 0
    with Station(StationConfig.load(a.config)) as st:
        for w in a.wafers:
            for d in range(1, a.dies + 1):
                m = measure_die(st, con, w, f"D{d:02d}", a.bank, heater_ring=None if a.no_heater else 1)
                n += 1
                n_pass += int(m["pass"])
    print(f"\nlot yield: {n_pass}/{n} = {100 * n_pass / n:.1f} %")
    print(db.dies(con).to_string(index=False))
