"""Measurement routines that combine station, analysis and database: what an operator runs."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import analysis, db
from .dut import design_resonances_nm
from .station import Station

# acceptance limits for the bank-2 (gap 0.31) design; the numbers a test engineer signs off
LIMITS = {
    "drop_loss_db": (None, 3.0),
    "fwhm_nm": (0.05, 0.60),
    "heater_ohm": (200.0, 400.0),
    "channel_spacing_nm": (1.2, 2.0),
    "res_offset_nm": (-3.9, 3.9),          # measured minus design, per ring: uniform over +-FSR/2, charted, not judged
    "tune_to_design_mw": (None, 200.0),    # heater power to bring ring 1 onto its design wavelength (red-shift only; a full FSR is ~176 mW)
    "loopback_dbm_1550": (-16.0, None),     # two 6 dB couplers + routing + fibers at 0 dBm in
}

SWEEP = dict(start_nm=1530.0, stop_nm=1570.0, step_nm=0.02)
HEATER_CURRENTS_A = [0.0, 4e-3, 8e-3, 12e-3, 16e-3]


def judge(metrics: dict) -> tuple[str, list[str]]:
    fails = []
    for k in range(1, 5):
        if metrics.get(f"ring{k}_found", 0) < 1:
            fails.append(f"ring{k}: no resonance near design")
            continue
        lo, hi = LIMITS["drop_loss_db"]
        if metrics[f"ring{k}_drop_loss_db"] > hi:
            fails.append(f"ring{k}: drop loss {metrics[f'ring{k}_drop_loss_db']:.2f} dB > {hi}")
        lo, hi = LIMITS["fwhm_nm"]
        if not (lo <= metrics[f"ring{k}_fwhm_nm"] <= hi):
            fails.append(f"ring{k}: FWHM {metrics[f'ring{k}_fwhm_nm']:.3f} nm outside [{lo}, {hi}]")
        lo, hi = LIMITS["heater_ohm"]
        r = metrics.get(f"heater{k}_ohm", float("nan"))
        if not (lo <= r <= hi):
            fails.append(f"heater{k}: {r:.0f} ohm outside [{lo}, {hi}]" if np.isfinite(r) else f"heater{k}: open")
    if "heater1_tune_to_design_mw" in metrics:
        lo, hi = LIMITS["tune_to_design_mw"]
        if metrics["heater1_tune_to_design_mw"] > hi:
            fails.append(f"heater1: {metrics['heater1_tune_to_design_mw']:.1f} mW to reach the design wavelength > {hi}")
    if "channel_spacing_nm" in metrics:
        lo, hi = LIMITS["channel_spacing_nm"]
        if not (lo <= metrics["channel_spacing_nm"] <= hi):
            fails.append(f"channel spacing {metrics['channel_spacing_nm']:.2f} nm")
    lo, hi = LIMITS["loopback_dbm_1550"]
    if metrics.get("loopback_dbm_1550", 0) < lo:
        fails.append(f"loopback {metrics['loopback_dbm_1550']:.1f} dBm: coupling or alignment problem")
    return ("PASS" if not fails else "FAIL"), fails


def measure_die(st: Station, con, wafer: str, die: str, bank: int = 2, heater_ring: int | None = 1,
                verbose: bool = True) -> dict:
    """Load a die, check heaters, sweep, analyse, tune one heater, store everything. Returns metrics."""
    st.load_die(wafer, die, bank)
    design = design_resonances_nm(bank)
    params = dict(bank=bank, **SWEEP, design_res_nm=design)

    # 1. heaters at 1 mA: resistance, open, short
    res_r = st.heater_resistances()
    metrics = {f"{k}_ohm": v for k, v in res_r.items()}

    # 2. spectrum, all six channels
    sweep = st.wavelength_sweep(**SWEEP)
    run_id = db.new_run(con, st.cfg.name, st.cfg.operator, wafer, die, bank, "wavelength_sweep", params)
    db.add_trace(con, run_id, sweep)
    metrics.update(analysis.die_summary(sweep, design))
    for k in range(1, 5):
        if f"ring{k}_res_nm" in metrics:
            metrics[f"ring{k}_res_offset_nm"] = metrics[f"ring{k}_res_nm"] - design[k]

    # 3. thermal tuning of one ring, only if its heater is sane
    if heater_ring and 50 < metrics.get(f"heater{heater_ring}_ohm", 0) < 1e5 and metrics.get(f"ring{heater_ring}_found"):
        r0 = metrics[f"ring{heater_ring}_res_nm"]
        hs = st.heater_sweep(heater_ring, HEATER_CURRENTS_A, r0 - 0.6, r0 + 5.0, 0.02)
        tune = analysis.heater_tuning(hs, f"drop{heater_ring}")
        hrun = db.new_run(con, st.cfg.name, st.cfg.operator, wafer, die, bank, "heater_sweep",
                          dict(ring=heater_ring, currents_a=HEATER_CURRENTS_A))
        db.add_trace(con, hrun, hs, aux_col="power_mw")
        if "nm_per_mw" in tune:
            metrics[f"heater{heater_ring}_tuning_nm_per_mw"] = tune["nm_per_mw"]
            fsr = metrics[f"ring{heater_ring}_fsr_nm"]
            metrics[f"heater{heater_ring}_ppi_mw"] = (fsr / 2) / tune["nm_per_mw"] if tune["nm_per_mw"] > 0 else float("nan")
            # heaters only red-shift: the shift needed is (-offset) modulo FSR
            shift = (-metrics[f"ring{heater_ring}_res_offset_nm"]) % fsr if np.isfinite(fsr) and fsr > 0 else float("nan")
            metrics[f"heater{heater_ring}_tune_to_design_mw"] = shift / tune["nm_per_mw"] if tune["nm_per_mw"] > 0 else float("nan")
            db.add_results(con, hrun, {k: v for k, v in metrics.items() if "tuning" in k or "ppi" in k})

    verdict, fails = judge(metrics)
    metrics["pass"] = 1.0 if verdict == "PASS" else 0.0
    db.add_results(con, run_id, metrics)
    db.set_die_status(con, wafer, die, verdict, "; ".join(fails))
    metrics["_run_id"], metrics["_verdict"], metrics["_fails"], metrics["_sweep"] = run_id, verdict, fails, sweep
    if verbose:
        print(f"{wafer} {die} bank{bank}: {verdict}" + (f"  [{'; '.join(fails)}]" if fails else "")
              + f"  ({sweep.attrs['seconds']} s sweep)")
    return metrics


def metrics_table(m: dict) -> pd.DataFrame:
    rows = []
    for k in range(1, 5):
        if m.get(f"ring{k}_found"):
            rows.append(dict(ring=k, res_nm=round(m[f"ring{k}_res_nm"], 3), offset_nm=round(m[f"ring{k}_res_offset_nm"], 2),
                             fwhm_nm=round(m[f"ring{k}_fwhm_nm"], 3), drop_loss_db=round(m[f"ring{k}_drop_loss_db"], 2),
                             fsr_nm=round(m[f"ring{k}_fsr_nm"], 2), heater_ohm=round(m.get(f"heater{k}_ohm", float("nan")), 0)))
        else:
            rows.append(dict(ring=k, res_nm=None, heater_ohm=round(m.get(f"heater{k}_ohm", float("nan")), 0)))
    return pd.DataFrame(rows)
