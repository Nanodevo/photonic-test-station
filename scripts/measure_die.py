"""Measure one die: heaters, spectrum, thermal tuning; store; print; optional figure.

    python scripts/measure_die.py --wafer W01 --die D07 --bank 2 --plot
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pts import db
from pts.routines import measure_die, metrics_table
from pts.station import Station, StationConfig


def plot_die(m: dict, out: Path):
    s = m["_sweep"]
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for c in ("drop1", "drop2", "drop3", "drop4"):
        ax.plot(s.wavelength_nm, s[c], lw=0.9, label=c)
    ax.plot(s.wavelength_nm, s.thru, lw=0.9, color="k", alpha=0.6, label="thru")
    ax.plot(s.wavelength_nm, s.loopback, lw=0.9, color="gray", ls="--", label="loopback")
    ax.set(xlabel="wavelength (nm)", ylabel="power (dBm)", title=f"{m['_verdict']}: {a.wafer} {a.die}, bank {a.bank}", ylim=(-70, 0))
    ax.legend(ncol=6, fontsize=8, loc="lower center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("figure:", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wafer", default="W01")
    ap.add_argument("--die", default="D01")
    ap.add_argument("--bank", type=int, default=2)
    ap.add_argument("--config", default="config/virtual.json")
    ap.add_argument("--db", default="data/results.sqlite")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    con = db.connect(a.db)
    with Station(StationConfig.load(a.config)) as st:
        print(st.idns())
        m = measure_die(st, con, a.wafer, a.die, a.bank)
    print(metrics_table(m).to_string(index=False))
    for k in ("heater1_tuning_nm_per_mw", "heater1_ppi_mw", "channel_spacing_nm", "loopback_dbm_1550", "thru_notch_depth_db"):
        if k in m:
            print(f"{k:28s} {m[k]:.3f}")
    if a.plot:
        m["_die"] = f" {a.wafer} {a.die}"
        plot_die(m, Path("docs/img") / f"die_{a.wafer}_{a.die}_bank{a.bank}.png")
