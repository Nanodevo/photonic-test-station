"""Trace -> numbers. Pure functions on DataFrames / arrays, testable without hardware."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_widths


def _lin(dbm):
    return 10 ** (np.asarray(dbm, dtype=float) / 10)


def resonances(lam_nm, p_dbm, prominence_db: float = 3.0, min_height_dbm: float = -60.0,
               rel_height_db: float = 15.0) -> pd.DataFrame:
    """Drop-port peaks: wavelength, peak level, 3 dB width (from the linear trace).

    A peak must sit within `rel_height_db` of the strongest one: the Lorentzian tails
    of a drop channel are carved by the other rings' through-notches and would
    otherwise count as resonances.
    """
    lam = np.asarray(lam_nm, float)
    p = np.asarray(p_dbm, float)
    idx, props = find_peaks(p, prominence=prominence_db, height=max(min_height_dbm, float(np.max(p)) - rel_height_db))
    if len(idx) == 0:
        return pd.DataFrame(columns=["wavelength_nm", "peak_dbm", "fwhm_nm", "prominence_db"])
    w, _, _, _ = peak_widths(_lin(p), idx, rel_height=0.5)
    step = np.median(np.diff(lam))
    return pd.DataFrame({
        "wavelength_nm": lam[idx],
        "peak_dbm": p[idx],
        "fwhm_nm": w * step,
        "prominence_db": props["prominences"],
    })


def notches(lam_nm, p_dbm, prominence_db: float = 3.0) -> pd.DataFrame:
    """Through-port notches: wavelength and depth."""
    lam = np.asarray(lam_nm, float)
    p = np.asarray(p_dbm, float)
    idx, props = find_peaks(-p, prominence=prominence_db)
    return pd.DataFrame({"wavelength_nm": lam[idx], "depth_db": props["prominences"]})


def fsr_nm(res_wavelengths) -> float:
    r = np.sort(np.asarray(res_wavelengths, float))
    return float(np.median(np.diff(r))) if len(r) > 1 else float("nan")


def strongest_near(res: pd.DataFrame, target_nm: float, window_nm: float) -> pd.Series | None:
    """The highest peak within +-window of a target wavelength (the design value)."""
    sel = res[(res.wavelength_nm - target_nm).abs() <= window_nm]
    return None if sel.empty else sel.loc[sel.peak_dbm.idxmax()]


def loss_vs_reference(lam_nm, p_dbm, ref_lam_nm, ref_dbm, at_nm: float) -> float:
    """Insertion loss of a path relative to the loopback at the same wavelength (removes coupler + fiber)."""
    p = np.interp(at_nm, np.asarray(lam_nm, float), np.asarray(p_dbm, float))
    r = np.interp(at_nm, np.asarray(ref_lam_nm, float), np.asarray(ref_dbm, float))
    return float(r - p)


def heater_tuning(hs: pd.DataFrame, channel_col: str) -> dict:
    """From a heater_sweep frame: resonance wavelength per power step, slope nm/mW, resistance."""
    rows = []
    for (i_a, v, p_mw), g in hs.groupby(["current_a", "voltage_v", "power_mw"], sort=True):
        res = resonances(g.wavelength_nm.values, g[channel_col].values)
        if res.empty:
            continue
        rows.append(dict(current_a=i_a, voltage_v=v, power_mw=p_mw, wavelength_nm=res.loc[res.peak_dbm.idxmax(), "wavelength_nm"]))
    t = pd.DataFrame(rows)
    out = {"points": t}
    if len(t) >= 2 and t.power_mw.max() > 0:
        slope, _ = np.polyfit(t.power_mw, t.wavelength_nm, 1)
        out["nm_per_mw"] = float(slope)
        nz = t[t.current_a > 0]
        out["resistance_ohm"] = float(np.median(nz.voltage_v / nz.current_a)) if not nz.empty else float("nan")
    return out


def die_summary(sweep: pd.DataFrame, design_res_nm: dict, window_nm: float = 4.0) -> dict:
    """Per-ring metrics from one wavelength sweep: resonance, width, drop loss vs loopback, through notch."""
    lam = sweep.wavelength_nm.values
    out = {}
    for k in range(1, 5):
        col = f"drop{k}"
        if col not in sweep:
            continue
        res = resonances(lam, sweep[col].values)
        pk = strongest_near(res, design_res_nm[k], window_nm)
        if pk is None:
            out[f"ring{k}_found"] = 0.0
            continue
        out[f"ring{k}_found"] = 1.0
        out[f"ring{k}_res_nm"] = float(pk.wavelength_nm)
        out[f"ring{k}_fwhm_nm"] = float(pk.fwhm_nm)
        out[f"ring{k}_drop_loss_db"] = loss_vs_reference(lam, sweep[col].values, lam, sweep["loopback"].values, float(pk.wavelength_nm))
        out[f"ring{k}_fsr_nm"] = fsr_nm(res.wavelength_nm.values)
    if "thru" in sweep:
        n = notches(lam, sweep["thru"].values)
        out["thru_notches"] = float(len(n))
        out["thru_notch_depth_db"] = float(n.depth_db.median()) if not n.empty else 0.0
    if all(f"ring{k}_res_nm" in out for k in range(1, 5)):
        r = np.array([out[f"ring{k}_res_nm"] for k in range(1, 5)])
        fsr = out["ring1_fsr_nm"]
        d = np.diff(r)
        if np.isfinite(fsr) and fsr > 0:
            d = d - fsr * np.round(d / fsr)      # the ladder wraps around the FSR
        out["channel_spacing_nm"] = float(np.mean(d))
    out["loopback_dbm_1550"] = float(np.interp(1550.0, lam, sweep["loopback"].values))
    return out
