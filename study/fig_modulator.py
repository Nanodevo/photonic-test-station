"""Modulator: Mach-Zehnder and ring, DC transfer, EO bandwidth, eye and BER."""
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from common import OUT, nrz_eye, draw_eye

Vpi = 4.0
V = np.linspace(-1.5 * Vpi, 1.5 * Vpi, 600)

def mzm_db(V, imbalance=1.0, phi0=0.0):
    e = 1 + imbalance * np.exp(1j * (np.pi * V / Vpi + phi0))
    T = np.abs(e) ** 2 / (1 + imbalance) ** 2
    return 10 * np.log10(np.clip(T, 1e-6, None))

fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.4))
a = ax[0, 0]
a.plot(V, mzm_db(V), label="balanced arms: ER > 30 dB")
a.plot(V, mzm_db(V, imbalance=0.8), label="arm imbalance 0.8: ER 19 dB")
a.plot(V, mzm_db(V, phi0=0.6), label="phase offset: bias point moved")
a.axvline(0, color="k", lw=0.6); a.axvline(Vpi, color="k", lw=0.6)
a.annotate("Vπ", xy=(Vpi / 2, -1.5), ha="center")
a.set(title="(a) MZM transfer vs voltage (DC sweep)", xlabel="voltage (V)", ylabel="transmission (dB)", ylim=(-40, 1))
a.legend(loc="lower left")

lam = np.linspace(1549.0, 1551.0, 1200)
def ring_thru_db(lam, l0, fwhm=0.08, er_db=18.0):
    x = (lam - l0) / (fwhm / 2)
    T = 1 - (1 - 10 ** (-er_db / 10)) / (1 + x ** 2)
    return 10 * np.log10(T)
a = ax[0, 1]
for v, c in [(0, "tab:blue"), (1, "tab:orange"), (2, "tab:green")]:
    a.plot(lam, ring_thru_db(lam, 1550.0 + 0.02 * v), color=c, label=f"{v} V  (20 pm/V, depletion)")
a.axvline(1549.97, color="k", ls=":", lw=0.8)
a.annotate("laser parked\non the slope", xy=(1549.97, -2), xytext=(1549.3, -6), arrowprops=dict(arrowstyle="->"))
a.set(title="(b) ring modulator: resonance shifts with bias", xlabel="wavelength (nm)", ylabel="through (dB)", ylim=(-20, 1))
a.legend(loc="lower right")

a = ax[0, 2]
Vr = np.linspace(-0.5, 3, 300)
for det, lab in [(-0.03, "laser 30 pm below resonance"), (-0.06, "60 pm below: less depth, more linear"), (0.03, "30 pm above: resonance sweeps through the laser")]:
    a.plot(Vr, ring_thru_db(1550.0 + det, 1550.0 + 0.02 * Vr), label=lab)
a.set(title="(c) ring: transmission vs voltage, laser fixed", xlabel="reverse bias (V)", ylabel="through (dB)")
a.legend(loc="lower right")

f = np.logspace(8, 11, 400)
a = ax[1, 0]
def eo(f, f_rc, f_ph=None):
    h = 1 / (1 + 1j * f / f_rc)
    if f_ph:
        h = h / (1 + 1j * f / f_ph)
    return 20 * np.log10(np.abs(h))
for f_rc, f_ph, lab in [(35e9, None, "MZM, RC-limited 35 GHz"), (40e9, 19e9, "ring Q 10k: photon lifetime 8 ps"), (40e9, 6e9, "ring Q 30k: photon lifetime 25 ps")]:
    s = eo(f, f_rc, f_ph); i3 = np.argmax(s < -3)
    a.semilogx(f / 1e9, s, label=f"{lab}: f3dB {f[i3] / 1e9:.0f} GHz")
a.axhline(-3, color="k", lw=0.8, ls="--")
a.set(title="(d) RF: electro-optic S21 (VNA + fast PD)", xlabel="frequency (GHz)", ylabel="EO S21 (dB)", ylim=(-20, 2), xlim=(0.1, 100))
a.legend(loc="lower left")

t, tr = nrz_eye(f3db=30e9, er_db=8, noise=0.03, overshoot=0.6)
draw_eye(ax[1, 1], t, tr, "(e) 25 Gb/s eye, ER 8 dB, with driver overshoot")
a = ax[1, 2]
Prx = np.linspace(-22, -8, 200)                      # dBm average received power
def ber(Prx_dbm, er_db, sigma=1.2e-6, R=0.9):
    P = 1e-3 * 10 ** (Prx_dbm / 10)
    r = 10 ** (er_db / 10)
    P1, P0 = 2 * P * r / (r + 1), 2 * P / (r + 1)
    Q = R * (P1 - P0) / (2 * sigma)
    return 0.5 * erfc(Q / np.sqrt(2))
for er in (15, 6, 3):
    a.semilogy(Prx, ber(Prx, er), label=f"ER {er} dB")
a.axhline(1e-12, color="k", ls="--", lw=0.8); a.axhline(2.4e-4, color="gray", ls=":", lw=0.8)
a.text(-21.8, 3e-4, "KP4 FEC threshold", fontsize=8, color="gray"); a.text(-21.8, 1.6e-12, "error-free", fontsize=8)
a.set(title="(f) BER vs received power: ER penalty", xlabel="average received power (dBm)", ylabel="BER", ylim=(1e-15, 1))
a.legend(loc="upper right")
fig.suptitle("Modulator, die level: DC transfer (a-c) sets the bias point; RF (d-f) decides the data rate", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "modulator.png")
print("wrote", OUT / "modulator.png")
