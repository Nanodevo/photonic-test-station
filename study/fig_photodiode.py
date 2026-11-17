"""Photodiode: what the standard DC and RF curves look like, and what a deviation means."""
import numpy as np
import matplotlib.pyplot as plt
from common import OUT, nrz_eye, draw_eye

kT = 0.0259
V = np.linspace(0, -6, 400)                        # reverse bias, volts (negative)

# (a) dark current: diffusion/generation + shunt + breakdown multiplication
def dark(V, I_gen=15e-9, rsh=np.inf, vbr=-12.0, m=4):
    Vr = -V
    gen = I_gen * np.sqrt(1 + Vr / 0.5)              # depletion width grows ~ sqrt(V)
    shunt = Vr / rsh
    M = 1.0 / np.clip(1 - (Vr / -vbr) ** m, 1e-6, None)
    return (gen + shunt) * M * (1 - np.exp(-Vr / kT))   # no current at zero bias

# (b) responsivity vs bias: depletion completes, then saturates
def resp(V, R_max=0.9, V0=0.3):
    return R_max * (1 - np.exp(-(-V + 0.2) / V0))

# (d) S21: RC pole and transit-time roll-off
f = np.logspace(8, 11, 400)
def s21(f, f_rc, tau):
    h_rc = 1 / (1 + 1j * f / f_rc)
    w = 2 * np.pi * f * tau
    h_tr = (1 - np.exp(-1j * w)) / (1j * w)
    return 20 * np.log10(np.abs(h_rc * h_tr))

fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.4))
a = ax[0, 0]
a.semilogy(-V, dark(V), label="good: 26 nA at 1 V, breakdown at 12 V")
a.semilogy(-V, dark(V, rsh=30e6), label="leaky: 30 MΩ shunt (edge or contact)")
a.semilogy(-V, dark(V, vbr=-4.5), label="early breakdown at 4.5 V (defect)")
a.set(title="(a) dark current vs bias (DC, light off)", xlabel="reverse bias (V)", ylabel="dark current (A)")
a.legend(loc="upper left")

a = ax[0, 1]
a.plot(-V, resp(V), label="fully depleted by 1 V")
a.plot(-V, resp(V, V0=1.5), label="doping too high: still rising at 3 V")
a.plot(-V, resp(V, R_max=0.55), label="low plateau: thin absorber or poor coupling")
a.set(title="(b) responsivity vs bias (DC, known power in)", xlabel="reverse bias (V)", ylabel="responsivity (A/W)", ylim=(0, 1))
a.legend(loc="lower right", fontsize=8)

a = ax[0, 2]
P = np.logspace(-2, 1.3, 200)                      # mW
Ip = 0.9 * P / (1 + P / 6.0)
a.loglog(P, 0.9 * P, "--", color="gray", label="ideal 0.9 A/W")
a.loglog(P, Ip, label="measured: saturates (space charge)")
p1 = P[np.argmin(np.abs(10 * np.log10(Ip / (0.9 * P)) + 1))]
a.axvline(p1, color="tab:red", ls=":", label=f"1 dB compression at {p1:.1f} mW")
a.set(title="(c) linearity: photocurrent vs optical power", xlabel="optical power (mW)", ylabel="photocurrent (mA)")
a.legend(loc="upper left")

a = ax[1, 0]
for f_rc, tau, lab in [(45e9, 8e-12, "RC 45 GHz, transit 8 ps"), (45e9, 25e-12, "thick absorber: transit 25 ps"), (12e9, 8e-12, "large pad / no de-embedding: RC 12 GHz")]:
    s = s21(f, f_rc, tau)
    i3 = np.argmax(s < -3)
    a.semilogx(f / 1e9, s, label=f"{lab}: f3dB {f[i3] / 1e9:.0f} GHz")
a.axhline(-3, color="k", lw=0.8, ls="--")
a.set(title="(d) RF: opto-electrical S21 (VNA)", xlabel="frequency (GHz)", ylabel="|S21| (dB, normalised)", ylim=(-20, 2), xlim=(0.1, 100))
a.legend(loc="center left", fontsize=7.5)

t, tr = nrz_eye(f3db=30e9, er_db=10, noise=0.03)
draw_eye(ax[1, 1], t, tr, "(e) RF: 25 Gb/s eye, 30 GHz diode")
t, tr = nrz_eye(f3db=11e9, er_db=10, noise=0.03)
draw_eye(ax[1, 2], t, tr, "(f) same data, 11 GHz diode: slower edges")
fig.suptitle("Photodiode, die level: DC curves (a-c) with needle probes and an SMU, RF curves (d-f) with GSG probes", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "photodiode.png")
print("wrote", OUT / "photodiode.png")
