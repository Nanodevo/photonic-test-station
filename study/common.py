"""Shared helpers for the study figures: styling, NRZ eye diagrams, simple filters."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "img" / "study"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 10.5, "axes.labelsize": 9.5, "legend.fontsize": 8.5,
                     "figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def lowpass_first_order(x, fs, f3db):
    """One-pole low-pass on a sampled waveform (fs samples/s)."""
    a = np.exp(-2 * np.pi * f3db / fs)
    y = np.zeros_like(x)
    for i in range(1, len(x)):
        y[i] = a * y[i - 1] + (1 - a) * x[i]
    return y


def nrz_eye(bitrate=25e9, f3db=30e9, er_db=8.0, n_bits=400, sps=32, noise=0.03, seed=1, overshoot=0.0):
    """Return (t_ui, traces) for an NRZ eye: 2-UI segments of a bandwidth-limited, noisy waveform."""
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, n_bits)
    fs = bitrate * sps
    x = np.repeat(bits, sps).astype(float)
    r = 10 ** (er_db / 10)
    lo, hi = 1.0 / r, 1.0
    x = lo + (hi - lo) * x
    y = lowpass_first_order(x, fs, f3db)
    if overshoot:
        # a second, faster pole with peaking: add a differentiated, delayed copy
        d = np.gradient(y) * overshoot * sps / 4
        y = y + d
    y = y + rng.normal(0, noise, len(y))
    seg = 2 * sps
    n = (len(y) - seg) // sps
    traces = np.array([y[i * sps: i * sps + seg] for i in range(2, n)])
    t = np.arange(seg) / sps
    return t, traces


def draw_eye(ax, t, traces, title):
    for tr in traces[::2]:
        ax.plot(t, tr, color="tab:blue", alpha=0.08, lw=0.8)
    ax.set(title=title, xlabel="time (UI)", ylabel="optical power (a.u.)", ylim=(-0.15, 1.3))
