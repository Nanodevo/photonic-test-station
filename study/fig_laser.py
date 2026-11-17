"""Laser diode (edge emitter): L-I-V, temperature, spectrum, modulation response, RIN."""
import numpy as np
import matplotlib.pyplot as plt
from common import OUT

I = np.linspace(0, 80, 400)            # mA
def liv(I, Ith0=15.0, eta=0.35, Rs=4.0, T0=60.0, Rth=60.0, Tamb=25.0, kink_at=None):
    """P in mW, V in volts, with self-heating raising the threshold (too weak here to show rollover)."""
    V = 0.9 + 0.026 * 2 * np.log1p(I / 0.01) + Rs * I * 1e-3
    P = np.zeros_like(I)
    for _ in range(30):                          # self-consistent temperature
        T = Tamb + Rth * (V * I * 1e-3 - P * 1e-3)
        Ith = Ith0 * np.exp((T - 25.0) / T0)
        P = np.clip(eta * (I - Ith), 0, None)
    if kink_at:
        P = np.where(I > kink_at, P * 0.93, P)
    return P, V, Ith

fig, ax = plt.subplots(2, 3, figsize=(12.5, 7.4))
a = ax[0, 0]
P, V, _ = liv(I)
a.plot(I, P, label="L-I curve")
Pk, _, _ = liv(I, kink_at=45)
a.plot(I, Pk, ls="--", label="kink at 45 mA: mode hop")
sel = (I > 20) & (I < 35)
m, b = np.polyfit(I[sel], P[sel], 1)
a.plot(I, m * I + b, ":", color="gray", label=f"fit: Ith = {-b / m:.1f} mA, η = {m:.2f} mW/mA")
a.set(title="(a) L-I-V (DC: SMU current, power meter)", xlabel="current (mA)", ylabel="power (mW)", ylim=(0, 25))
a.legend(loc="upper left", fontsize=7.5)
a2 = a.twinx(); a2.plot(I, V, color="tab:red", lw=1); a2.set_ylabel("voltage (V)", color="tab:red"); a2.set_ylim(0, 3); a2.grid(False)
dv = np.gradient(V, I)[-1] * 1e3
a2.text(26, 0.25, f"dV/dI at 80 mA: {dv:.1f} Ω", color="tab:red", fontsize=8.5)

a = ax[0, 1]
for T in (25, 45, 65):
    P, _, Ith = liv(I, Tamb=T)
    a.plot(I, P, label=f"{T} °C: Ith {Ith[0]:.0f} mA")
a.set(title="(b) L-I vs temperature (chuck): T0 = 60 K", xlabel="current (mA)", ylabel="power (mW)", ylim=(0, 25))
a.legend(loc="upper left")

a = ax[0, 2]
lam = np.linspace(1547, 1553, 3000)
def lorentz(lam, l0, w):
    """A line as an analyser with resolution w draws it: Gaussian filter shape on a -65 dB floor."""
    return np.exp(-4 * np.log(2) * ((lam - l0) / w) ** 2) + 10 ** (-6.5)
dfb = lorentz(lam, 1550.0, 0.02) + 10 ** (-4.5) * (lorentz(lam, 1549.2, 0.02) + lorentz(lam, 1550.8, 0.02))
fp = sum(np.exp(-((1550.0 + 0.8 * k) - 1550.0) ** 2 / 2.0) * lorentz(lam, 1550.0 + 0.8 * k, 0.02) for k in range(-4, 5))
a.plot(lam, 10 * np.log10(dfb / dfb.max()), label="DFB: single mode, SMSR 45 dB")
a.plot(lam, 10 * np.log10(fp / fp.max()) - 0.0, alpha=0.7, label="Fabry-Perot: mode comb, 0.8 nm spacing")
a.annotate("SMSR 45 dB", xy=(1550.8, -45), xytext=(1551.0, -52), fontsize=8)
a.set(title="(c) spectrum (OSA, 20 pm)", xlabel="wavelength (nm)", ylabel="relative power (dB)", ylim=(-70, 32))
a.legend(loc="upper left", fontsize=8)

f = np.linspace(0.1e9, 30e9, 600)
def resp_db(f, Ib, Ith=15.0, D=2.0e9, K=0.25e-9, g0=1e9):
    fr = D * np.sqrt(max(Ib - Ith, 0.1))
    g = K * fr ** 2 + g0
    H = fr ** 4 / ((fr ** 2 - f ** 2) ** 2 + (g * f / (2 * np.pi)) ** 2)
    return 10 * np.log10(H / H[0]), fr
a = ax[1, 0]
for Ib in (20, 40, 70):
    h, fr = resp_db(f, Ib); i3 = np.argmax(h < -3)
    a.plot(f / 1e9, h, label=f"{Ib} mA: fr {fr / 1e9:.1f} GHz, f3dB {f[i3] / 1e9:.0f} GHz")
a.axhline(-3, color="k", ls="--", lw=0.8)
a.set(title="(d) RF: modulation response (VNA + fast PD)", xlabel="frequency (GHz)", ylabel="response (dB)", ylim=(-25, 12))
a.legend(loc="lower left")

a = ax[1, 1]
for Ib, level in ((20, -140), (40, -150), (70, -155)):
    h, fr = resp_db(f, Ib)
    a.plot(f / 1e9, level + h, label=f"{Ib} mA")
a.set(title="(e) RF: relative intensity noise (ESA)", xlabel="frequency (GHz)", ylabel="RIN (dB/Hz)", ylim=(-165, -120))
a.legend(loc="upper right")

a = ax[1, 2]
Pg, _, _ = liv(I); Pk2, _, _ = liv(I, kink_at=45)
a.plot(I[1:], np.gradient(Pg, I)[1:], label="good: flat above threshold")
a.plot(I[1:], np.gradient(Pk2, I)[1:], ls="--", label="kink: a dip in the slope at 45 mA")
a.set(title="(f) dP/dI finds the kinks", xlabel="current (mA)", ylabel="slope efficiency (mW/mA)", ylim=(-0.1, 0.5))
a.legend(loc="center right", fontsize=8)
fig.suptitle("Laser diode, die level: L-I-V and spectrum at DC, modulation response and RIN at RF", y=0.995)
fig.tight_layout()
fig.savefig(OUT / "laser.png")
print("wrote", OUT / "laser.png")
