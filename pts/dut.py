"""Physics model of the device under test: the WDM4 ring bank.

The chip is the cornerstone-pic tape-out: CORNERSTONE SOI, 450 x 220 nm strip
waveguides, racetrack add-drop rings (corner radius 10 um, length_x = 4 um,
length_y = 0.6 + 0.069 k um), laid out three times with coupling gaps
0.26 / 0.31 / 0.36 um (banks 1-3), one TiN heater per ring, grating couplers on
a 127 um pitch, one reference loopback. Waveguide numbers (n_eff, n_g,
coupling versus gap) are the femwell results in data/strip_450x220.json.

The ring's four corners are the kit's Euler bends, not quarter circles: each is
16.637 um long, and the coupling is integrated along the corner's real shape
(cornerstone-pic/wdm_models.py, November 2026 correction of that design).

Per-die process variation is seeded from (wafer, die), so a die measures the
same every time it is loaded, and some dies carry a defect. That is what makes
the virtual station worth running: the analysis has to find them.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data" / "strip_450x220.json"
_d = json.loads(DATA.read_text())
NEFF0, NG, WL0_UM = _d["neff0"], _d["ng"], _d["wl0"]
KAPPA0, G0, GAMMA = _d["kappa0"], _d["g0"], _d["gamma"]

# layout
R_UM, LENGTH_X, LENGTH_Y0, DL_Y = 10.0, 4.0, 0.6, 0.069
L_CORNER_UM = 16.637       # one Euler corner of the kit's ring cell (a quarter circle would be 15.708)
LEFF_CORNER_UM = 2.718     # integral of exp(-gamma * distance from the bus) along one corner, gamma from the data file
BANK_GAPS = {1: 0.26, 2: 0.31, 3: 0.36}
N_RINGS = 4

# nominal process and packaging numbers
ALPHA_DB_CM = 3.0          # propagation loss of the straight strip (assumed; the foundry's library has no strip page)
BEND_DB = 0.0116           # loss of one 90 degree bend at R = 10 um, 450 nm strip: cut-back slope in CORNERSTONE's
                           # SOI standard components library (Mar 2023); the whole loss of the bend, charged once per corner
DNEFF_DW = 0.00219         # d n_eff / d width, per nm (450 x 220 strip, TE, 1550 nm; femwell width sweep, cornerstone-pic step 18)
DNEFF_DT = 0.0035          # d n_eff / d thickness, per nm
P_PI_MW = 88.0             # electrical heater power for a pi phase shift: the foundry's qualification bound of 30 mW per pi
                           # for the heated waveguide, times 113/38.5 because only 38.5 um of the 113 um heater lies over the ring
R_HEATER = 300.0           # TiN heater resistance, ohm
GC_LOSS_DB = 6.0           # grating coupler loss at its centre wavelength (kit model; the library measures 5.5 to 6.5 dB)
GC_1DB_HALF_BW_NM = 17.5   # 1 dB bandwidth = 35 nm
ROUTE_LOSS_DB = 0.3        # on-chip routing per path
NOISE_FLOOR_DBM = -70.0
PM_REPEAT_DB = 0.02        # power-meter repeatability

# power-meter channels (one bank in front of the fiber array at a time)
CHANNELS = {1: "drop1", 2: "drop2", 3: "drop3", 4: "drop4", 5: "thru", 6: "loopback"}


@dataclass
class DieParams:
    wafer: str
    die: str
    dw_nm: float                  # waveguide width error, die-wide
    dt_nm: float                  # SOI thickness error, wafer map + die residual
    dgap_nm: float                # coupling-gap error
    alpha_db_cm: float
    gc_loss_db: float
    gc_center_nm: float
    ring_offset_nm: np.ndarray    # 12 rings: residual resonance offset
    heater_r: np.ndarray          # 12 heaters, ohm (1e9 = open, ~3 = short)
    ring_extra_db_cm: np.ndarray  # 12 rings: excess loss (particle)
    defects: list[str]


def _seed(*parts: str) -> int:
    return int(hashlib.sha256("/".join(parts).encode()).hexdigest()[:8], 16)


def die_params(wafer: str, die: str) -> DieParams:
    """Deterministic process variation for one die."""
    wrng = np.random.default_rng(_seed(wafer))
    rng = np.random.default_rng(_seed(wafer, die))
    dt = wrng.normal(0.0, 2.0) + rng.normal(0.0, 1.0)   # thickness: mostly wafer-level
    dw = rng.normal(0.0, 6.0)                            # width: die-level litho/etch
    dgap = rng.normal(0.0, 8.0)
    alpha = max(1.0, rng.normal(ALPHA_DB_CM, 0.5))
    gc = rng.normal(GC_LOSS_DB, 0.4)
    gcc = rng.normal(1550.0, 3.0)
    offsets = rng.normal(0.0, 0.15, 3 * N_RINGS)
    heater_r = rng.normal(R_HEATER, 15.0, 3 * N_RINGS)
    extra = np.zeros(3 * N_RINGS)
    defects: list[str] = []
    u = rng.random()                                     # about 8 % defective dies
    k = int(rng.integers(3 * N_RINGS))
    if u < 0.04:
        extra[k] = 40.0
        defects.append(f"bank{k // N_RINGS + 1} ring{k % N_RINGS + 1}: particle, excess loss")
    elif u < 0.06:
        heater_r[k] = 1e9
        defects.append(f"bank{k // N_RINGS + 1} heater{k % N_RINGS + 1}: open")
    elif u < 0.08:
        heater_r[k] = 3.0
        defects.append(f"bank{k // N_RINGS + 1} heater{k % N_RINGS + 1}: short")
    return DieParams(wafer, die, dw, dt, dgap, alpha, gc, gcc, offsets, heater_r, extra, defects)


# ---------------------------------------------------------------- ring physics
def ring_length_um(k: int) -> float:
    return 4 * L_CORNER_UM + 2 * LENGTH_X + 2 * (LENGTH_Y0 + DL_Y * k)


def ring_straight_um(k: int) -> float:
    return 2 * LENGTH_X + 2 * (LENGTH_Y0 + DL_Y * k)


def coupling_k(gap_um: float) -> float:
    """Power cross-coupling per pass, sin^2(kappa(gap) * l_eff)."""
    l_eff = LENGTH_X + 2 * LEFF_CORNER_UM
    kappa = KAPPA0 * np.exp(-GAMMA * (gap_um - G0))
    return float(np.sin(kappa * l_eff) ** 2)


def round_trip_loss_db(k: int, alpha_db_cm: float = ALPHA_DB_CM, extra_db_cm: float = 0.0) -> float:
    """Power loss of one round trip in dB: four corners at the foundry's bend figure, the straights
    at the strip loss. A die whose strip loss is off nominal has its bend loss scaled alike (assumed:
    both come from the same sidewalls). A particle adds its excess loss over the whole ring."""
    scale = alpha_db_cm / ALPHA_DB_CM
    return 4 * BEND_DB * scale + alpha_db_cm * ring_straight_um(k) * 1e-4 + extra_db_cm * ring_length_um(k) * 1e-4


def n_eff(lam_um, dn: float = 0.0):
    """First-order dispersion around 1550 nm: n_eff(l) = n_eff0 - (n_g - n_eff0)(l - l0)/l0."""
    return NEFF0 + dn - (NG - NEFF0) * (lam_um - WL0_UM) / WL0_UM


def add_drop(lam_um, L_um: float, K: float, loss_db: float, dn: float, phi_heater: float):
    """Symmetric add-drop ring. Returns (drop, through) power transmission.

    t = sqrt(1-K) is the field through-coupling, A the round-trip field
    attenuation for a round-trip power loss of loss_db, phi the round-trip phase.
    """
    t = np.sqrt(1.0 - K)
    A = 10 ** (-loss_db / 20.0)
    phi = 2 * np.pi * n_eff(lam_um, dn) * L_um / lam_um + phi_heater
    e = np.exp(1j * phi)
    D = 1.0 - t * t * A * e
    drop = -K * np.sqrt(A) * np.exp(1j * phi / 2) / D
    thru = (t - t * A * e) / D
    return np.abs(drop) ** 2, np.abs(thru) ** 2


class DUT:
    """The chip on the chuck: which die, which bank is in front of the fiber array,
    the fiber alignment of the moment, and the heater currents."""

    def __init__(self, rng_seed: int = 0):
        self.rng = np.random.default_rng(rng_seed)
        self.params: DieParams | None = None
        self.bank = 2
        self.align_db = np.zeros(len(CHANNELS) + 1)   # one per output fiber, last entry: the input fiber
        self.heater_ma = np.zeros(N_RINGS)
        self.compliance_v = 10.0

    # ---- prober side
    def load(self, wafer: str, die: str):
        self.params = die_params(wafer, die)
        self.heater_ma[:] = 0.0
        self.realign()

    def realign(self):
        """A fresh fiber-array alignment: a new random excess loss per fiber."""
        self.align_db = np.abs(self.rng.normal(0.0, 0.3, len(CHANNELS) + 1))

    # ---- heaters (SMU side)
    def _hidx(self, ring: int) -> int:
        return (self.bank - 1) * N_RINGS + (ring - 1)

    def heater_resistance(self, ring: int) -> float:
        return float(self.params.heater_r[self._hidx(ring)])

    def heater_voltage(self, ring: int) -> float:
        """Voltage the SMU sees for the programmed current, clipped at compliance."""
        r = self.heater_resistance(ring)
        return float(min(self.heater_ma[ring - 1] * 1e-3 * r, self.compliance_v))

    def heater_current_actual(self, ring: int) -> float:
        """Current that actually flows (an open heater sits at compliance with ~0 A)."""
        r = self.heater_resistance(ring)
        return float(min(self.heater_ma[ring - 1] * 1e-3, self.compliance_v / r))

    def heater_power_mw(self, ring: int) -> float:
        return 1e3 * self.heater_voltage(ring) * self.heater_current_actual(ring)

    # ---- optics
    def _gc_db(self, lam_nm):
        p = self.params
        return -(p.gc_loss_db + ((lam_nm - p.gc_center_nm) / GC_1DB_HALF_BW_NM) ** 2)

    def transmissions(self, lam_nm):
        """Linear power transmission of every channel for the selected bank, chip only."""
        p = self.params
        lam_um = np.asarray(lam_nm, dtype=float) * 1e-3
        gap = BANK_GAPS[self.bank] + p.dgap_nm * 1e-3
        K = coupling_k(gap)
        dn_die = DNEFF_DW * p.dw_nm + DNEFF_DT * p.dt_nm
        out = {}
        bus = np.ones_like(lam_um)
        for k in range(N_RINGS):
            idx = (self.bank - 1) * N_RINGS + k
            dn = dn_die + p.ring_offset_nm[idx] * 1e-3 * NG / WL0_UM
            loss = round_trip_loss_db(k, p.alpha_db_cm, p.ring_extra_db_cm[idx])
            phi_h = np.pi * self.heater_power_mw(k + 1) / P_PI_MW
            d, t = add_drop(lam_um, ring_length_um(k), K, loss, dn, phi_h)
            out[f"drop{k + 1}"] = bus * d
            bus = bus * t
        out["thru"] = bus
        out["loopback"] = np.ones_like(lam_um)
        return out

    def power_dbm(self, channel: int, lam_nm: float, laser_dbm: float, laser_on: bool) -> float:
        """What the power meter reads on one channel."""
        floor = 10 ** (NOISE_FLOOR_DBM / 10)
        if not laser_on or self.params is None:
            return float(10 * np.log10(floor * self.rng.uniform(0.5, 1.5)))
        name = CHANNELS[channel]
        T = self.transmissions(lam_nm)[name]
        route = ROUTE_LOSS_DB        # same routing loss on every path, so the loopback removes it
        p_dbm = (laser_dbm + 2 * self._gc_db(lam_nm) - route + 10 * np.log10(max(float(T), 1e-30))
                 - self.align_db[-1] - self.align_db[channel - 1])
        p_lin = 10 ** (p_dbm / 10) + floor * self.rng.uniform(0.5, 1.5)
        return float(10 * np.log10(p_lin) + self.rng.normal(0.0, PM_REPEAT_DB))


def nominal_die() -> DieParams:
    """A die with zero process variation: the design as drawn."""
    z = np.zeros(3 * N_RINGS)
    return DieParams("NOMINAL", "D00", 0.0, 0.0, 0.0, ALPHA_DB_CM, GC_LOSS_DB, 1550.0, z.copy(),
                     np.full(3 * N_RINGS, R_HEATER), z.copy(), [])


def design_resonances_nm(bank: int = 2, near_nm: float = 1550.0) -> dict:
    """Design resonance of each ring closest to `near_nm` (what the analysis looks for)."""
    d = DUT()
    d.params = nominal_die()
    d.bank = bank
    d.align_db[:] = 0.0
    lam = np.arange(near_nm - 6.0, near_nm + 6.0, 0.005)
    tr = d.transmissions(lam)
    out = {}
    for k in range(1, N_RINGS + 1):
        y = tr[f"drop{k}"]
        i = int(np.argmax(y - 1e-3 * np.abs(lam - near_nm)))  # strongest peak, tie-break nearest
        out[k] = float(lam[i])
    return out
