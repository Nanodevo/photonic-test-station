import numpy as np

from pts import dut


def test_fsr_near_8nm_and_coupling():
    K = dut.coupling_k(0.31)
    assert 0.04 < K < 0.08
    lam = np.arange(1530, 1570, 0.005)
    d, t = dut.add_drop(lam * 1e-3, dut.ring_length_um(0), K, dut.round_trip_loss_db(0), 0.0, 0.0)
    peaks = lam[(d > np.roll(d, 1)) & (d > np.roll(d, -1)) & (d > 0.5)]
    assert 7.4 < np.median(np.diff(peaks)) < 7.9
    assert abs(dut.ring_length_um(0) - 75.75) < 0.01


def test_die_params_are_deterministic_and_defects_exist():
    a, b = dut.die_params("W01", "D03"), dut.die_params("W01", "D03")
    assert a.dw_nm == b.dw_nm and np.all(a.heater_r == b.heater_r)
    n_def = sum(bool(dut.die_params("W01", f"D{i:02d}").defects) for i in range(1, 101))
    assert 2 <= n_def <= 20


def test_open_heater_hits_compliance():
    d = dut.DUT()
    d.load("W01", "D01")
    d.params.heater_r[4] = 1e9          # bank 2, ring 1
    d.bank = 2
    d.heater_ma[0] = 5.0
    assert d.heater_voltage(1) == d.compliance_v
    assert d.heater_current_actual(1) < 1e-6
