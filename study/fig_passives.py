"""Passives: grating coupler spectrum, cutback, ring parameters, the reference plane (loopback)."""
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from common import OUT

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pts import dut

fig, ax = plt.subplots(2, 2, figsize=(12.5, 7.6))
lam = np.linspace(1500, 1600, 800)
def gc_db(lam, il=4.0, lc=1550.0, hw=17.5):
    return -(il + ((lam - lc) / hw) ** 2)
a = ax[0, 0]
a.plot(lam, 2 * gc_db(lam), label="pair as designed: 4.0 dB each, centre 1550 nm")
a.plot(lam, 2 * gc_db(lam, lc=1542.0), label="centre at 1542 nm: etch depth / thickness off")
a.plot(lam, 2 * gc_db(lam, il=5.5), label="5.5 dB each: fiber angle or polarisation")
a.axhline(-9, color="k", ls=":", lw=0.8); a.text(1501, -8.7, "1 dB below the peak", fontsize=8)
a.set(title="(a) grating-coupler pair, loopback structure (tunable laser + power meter)", xlabel="wavelength (nm)", ylabel="transmission (dB)", ylim=(-20, -6))
a.legend(loc="lower center")

a = ax[0, 1]
rng = np.random.default_rng(3)
L = np.array([0.5, 1.0, 2.0, 4.0])                    # cm
il = 8.3 + 2.7 * L + rng.normal(0, 0.25, 4)
m, b = np.polyfit(L, il, 1)
a.plot(L, il, "o", label="spirals of four lengths")
a.plot([0, 4.5], [b, b + 4.5 * m], "--", label=f"fit: {m:.2f} dB/cm, intercept {b:.1f} dB = two couplers + routing")
a.set(title="(b) cutback: loss vs length separates propagation from coupling", xlabel="spiral length (cm)", ylabel="insertion loss (dB)", xlim=(0, 4.5))
a.legend(loc="upper left")

a = ax[1, 0]
d = dut.DUT(); d.params = dut.nominal_die(); d.bank = 2; d.align_db[:] = 0
lam2 = np.arange(1544, 1562, 0.005)
tr = d.transmissions(lam2)
drop, thru = 10 * np.log10(tr["drop1"]), 10 * np.log10(tr["thru"])
a.plot(lam2, drop, label="drop 1"); a.plot(lam2, thru, color="k", alpha=0.7, label="through")
i0 = int(np.argmax(drop)); pk = lam2[i0]
lo = i0
while lo > 0 and drop[lo - 1] > drop.max() - 3: lo -= 1
hi = i0
while hi < len(drop) - 1 and drop[hi + 1] > drop.max() - 3: hi += 1
fwhm = lam2[hi] - lam2[lo]
a.annotate(f"FWHM {fwhm * 1e3:.0f} pm, Q = λ/FWHM ≈ {pk / fwhm:.0f}", xy=(pk, drop.max()), xytext=(pk - 7.2, -5), arrowprops=dict(arrowstyle="->"), fontsize=8.5)
a.annotate("", xy=(pk + 7.66, -14), xytext=(pk, -14), arrowprops=dict(arrowstyle="<->"))
a.text(pk + 3.83, -13, "FSR 7.7 nm", ha="center", fontsize=8.5)
a.set(title="(c) ring resonator: FSR, width, drop loss, through extinction", xlabel="wavelength (nm)", ylabel="transmission (dB)", ylim=(-35, 1))
a.legend(loc="lower left")

a = ax[1, 1]
d.params = dut.die_params("W01", "D09"); d.realign()
lam3 = np.arange(1530, 1570, 0.02)
raw = np.array([d.power_dbm(1, l, 0.0, True) for l in lam3])
lb = np.array([d.power_dbm(6, l, 0.0, True) for l in lam3])
a.plot(lam3, raw, lw=0.8, label="drop 1 as read on the power meter")
a.plot(lam3, lb, lw=0.8, color="gray", label="loopback (two couplers, no device)")
a.plot(lam3, raw - lb, lw=0.8, color="tab:green", label="drop 1 minus loopback: the device alone")
a.set(title="(d) the reference plane: subtract the loopback to remove couplers and fibers", xlabel="wavelength (nm)", ylabel="power (dBm) / loss (dB)", ylim=(-50, 2))
a.legend(loc="lower center")
fig.suptitle("Passives, die level: everything is an insertion-loss spectrum; the art is in the reference", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "passives.png")
print("wrote", OUT / "passives.png")
