"""Control charts over the measured dies: plasmafab I-MR charts, capability, out-of-control list.

    python scripts/spc_report.py --db data/results.sqlite --out docs/img/spc_lot.png
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from plasmafab.spc import capability, imr_chart

from pts import db
from pts.routines import LIMITS

METRICS = [
    ("channel_spacing_nm", "channel spacing (nm)", LIMITS["channel_spacing_nm"]),
    ("ring1_drop_loss_db", "ring 1 drop loss vs loopback (dB)", LIMITS["drop_loss_db"]),
    ("ring1_fwhm_nm", "ring 1 FWHM (nm)", LIMITS["fwhm_nm"]),
    ("heater1_ohm", "heater 1 resistance (ohm)", LIMITS["heater_ohm"]),
    ("loopback_dbm_1550", "loopback at 1550 nm (dBm)", LIMITS["loopback_dbm_1550"]),
    ("thru_notch_depth_db", "through-port notch depth (dB)", (10.0, None)),
]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/results.sqlite")
    ap.add_argument("--out", default="docs/img/spc_lot.png")
    ap.add_argument("--baseline", type=int, default=15)
    a = ap.parse_args()
    con = db.connect(a.db)
    w = db.results_wide(con)
    if w.empty:
        raise SystemExit("no results yet: run measure_lot.py first")
    w["label"] = w.wafer + "/" + w.die
    # The resonance offset and the heater power to reach the design wavelength are not charted: both are
    # wrapped modulo one free spectral range and spread evenly over it, so an individuals chart of them
    # cannot signal anything. They stay in the database.
    fig, axes = plt.subplots(3, 2, figsize=(12, 8.8))
    print(f"{'metric':28s} {'mean':>9s} {'sigma':>8s} {'Cpk':>6s}  out-of-control dies")
    for ax, (m, title, (lsl, usl)) in zip(axes.ravel(), METRICS):
        if m not in w:
            ax.set_visible(False); continue
        x = w[m].reset_index(drop=True)
        ok = x.notna() & np.isfinite(x)
        c = imr_chart(x[ok], baseline_n=a.baseline)
        cap = capability(x[ok], lsl, usl)
        ooc = w.loc[ok & ((x > c["ucl"]) | (x < c["lcl"])), "label"].tolist()
        spec = w.loc[ok & (((x > usl) if usl is not None else False) | ((x < lsl) if lsl is not None else False)), "label"].tolist()
        i = np.arange(len(x))
        ax.plot(i[ok], x[ok], "o-", ms=3, lw=0.8)
        ax.axhline(c["center"], color="k", lw=0.8)
        ax.axhline(c["ucl"], color="r", ls="--", lw=0.8); ax.axhline(c["lcl"], color="r", ls="--", lw=0.8)
        for lim in (lsl, usl):
            if lim is not None:
                ax.axhline(lim, color="orange", ls=":", lw=1.0)
        bad = ok & ((x > c["ucl"]) | (x < c["lcl"]))
        ax.plot(i[bad], x[bad], "rx", ms=8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("die #"); ax.grid(alpha=0.3)
        cpk = cap["cpk"]
        print(f"{m:28s} {cap['mean']:9.3f} {cap['sigma']:8.3f} {('%.2f' % cpk) if cpk is not None else '  -':>6s}  {', '.join(ooc) or '-'}"
              + (f"   [outside spec: {', '.join(spec)}]" if spec else ""))
    fig.suptitle("WDM4 bank 2, individuals charts (limits frozen on the baseline; orange = spec limits)", fontsize=11)
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=130)
    print("figure:", a.out)
