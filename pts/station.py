"""The station: instruments wired together, and the measurement routines.

Every routine returns a pandas DataFrame with the raw trace; the analysis
module turns traces into numbers; the db module stores both.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyvisa

from .instruments import SMU, PowerMeter, Prober, TunableLaser

CHANNELS = (1, 2, 3, 4, 5, 6)
CHANNEL_NAMES = {1: "drop1", 2: "drop2", 3: "drop3", 4: "drop4", 5: "thru", 6: "loopback"}


@dataclass
class StationConfig:
    name: str
    operator: str
    laser: str
    power_meter: str
    smu: str
    prober: str
    heater_compliance_v: float = 10.0

    @classmethod
    def load(cls, path: str | Path) -> "StationConfig":
        return cls(**json.loads(Path(path).read_text()))


class Station:
    def __init__(self, cfg: StationConfig):
        self.cfg = cfg
        self.rm = pyvisa.ResourceManager("@py")
        self.laser = TunableLaser(cfg.laser, self.rm)
        self.pm = PowerMeter(cfg.power_meter, self.rm)
        self.smu = SMU(cfg.smu, self.rm)
        self.prober = Prober(cfg.prober, self.rm)
        for ch in range(1, 5):
            self.smu.set_compliance(ch, cfg.heater_compliance_v)
        self.die = ("NONE", "NONE")
        self.bank = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            self.laser.output(False)
            for ch in range(1, 5):
                self.smu.set_current(ch, 0.0)
                self.smu.output(ch, False)
        finally:
            for i in (self.laser, self.pm, self.smu, self.prober):
                i.close()

    def idns(self) -> dict:
        return {"laser": self.laser.idn(), "power_meter": self.pm.idn(), "smu": self.smu.idn(), "prober": self.prober.idn()}

    # ---------------------------------------------------------------- setup
    def load_die(self, wafer: str, die: str, bank: int = 2) -> None:
        self.prober.load_die(wafer, die)
        self.prober.select_bank(bank)
        self.prober.align()
        if not self.prober.contact():
            raise RuntimeError("no contact after loading die")
        self.die, self.bank = (wafer, die), bank

    def heaters_off(self) -> None:
        for ch in range(1, 5):
            self.smu.set_current(ch, 0.0)
            self.smu.output(ch, False)

    # ---------------------------------------------------------------- sweeps
    def wavelength_sweep(self, start_nm: float, stop_nm: float, step_nm: float,
                         channels=CHANNELS, laser_dbm: float = 0.0) -> pd.DataFrame:
        """Step the laser, read every channel at each step. Columns: wavelength_nm, drop1..loopback (dBm)."""
        lams = np.round(np.arange(start_nm, stop_nm + step_nm / 2, step_nm), 4)
        self.laser.set_power_dbm(laser_dbm)
        self.laser.output(True)
        self.pm.set_cal_wavelength(float(np.mean(lams)))
        rows = []
        t0 = time.time()
        for lam in lams:
            self.laser.set_wavelength(float(lam))
            rows.append([lam] + self.pm.read_all_dbm(channels))
        self.laser.output(False)
        df = pd.DataFrame(rows, columns=["wavelength_nm"] + [CHANNEL_NAMES[c] for c in channels])
        df.attrs.update(dict(kind="wavelength_sweep", start_nm=start_nm, stop_nm=stop_nm, step_nm=step_nm,
                             laser_dbm=laser_dbm, seconds=round(time.time() - t0, 2)))
        return df

    def heater_resistances(self, i_test_a: float = 1e-3) -> dict:
        """Two-wire resistance check at a small test current, heater by heater (lead resistance included)."""
        out = {}
        for ch in range(1, 5):
            self.smu.set_current(ch, i_test_a)
            self.smu.output(ch, True)
            v, i = self.smu.measure_voltage(ch), self.smu.measure_current(ch)
            self.smu.set_current(ch, 0.0)
            self.smu.output(ch, False)
            out[f"heater{ch}"] = v / i if i > 0 else float("inf")
        return out

    def heater_sweep(self, ring: int, currents_a, start_nm: float, stop_nm: float, step_nm: float,
                     channel: int | None = None, laser_dbm: float = 0.0) -> pd.DataFrame:
        """For each heater current: a narrow wavelength sweep on the ring's drop channel.

        Returns one row per (current, wavelength). Analysis fits resonance vs power.
        """
        channel = channel or ring
        frames = []
        self.smu.output(ring, True)
        for i_a in currents_a:
            self.smu.set_current(ring, float(i_a))
            v, i_meas = self.smu.measure_voltage(ring), self.smu.measure_current(ring)
            df = self.wavelength_sweep(start_nm, stop_nm, step_nm, channels=(channel,), laser_dbm=laser_dbm)
            df.insert(0, "current_a", float(i_a))
            df.insert(1, "voltage_v", v)
            df.insert(2, "power_mw", 1e3 * v * i_meas)
            frames.append(df)
        self.smu.set_current(ring, 0.0)
        self.smu.output(ring, False)
        out = pd.concat(frames, ignore_index=True)
        out.attrs.update(dict(kind="heater_sweep", ring=ring, channel=channel, start_nm=start_nm,
                              stop_nm=stop_nm, step_nm=step_nm, laser_dbm=laser_dbm))
        return out
