import numpy as np

from pts import analysis


def lorentz_dbm(lam, centers, fwhm, peak_dbm=-10.0, floor_dbm=-50.0):
    lin = 10 ** (floor_dbm / 10) * np.ones_like(lam)
    for c in centers:
        lin = lin + 10 ** (peak_dbm / 10) / (1 + ((lam - c) / (fwhm / 2)) ** 2)
    return 10 * np.log10(lin)


def test_resonances_find_width_and_fsr():
    lam = np.arange(1540, 1560, 0.01)
    p = lorentz_dbm(lam, [1545.0, 1553.0], fwhm=0.2)
    r = analysis.resonances(lam, p)
    assert len(r) == 2
    assert abs(r.fwhm_nm.iloc[0] - 0.2) < 0.02
    assert abs(analysis.fsr_nm(r.wavelength_nm) - 8.0) < 0.02


def test_loss_vs_reference():
    lam = np.arange(1540, 1560, 0.1)
    assert abs(analysis.loss_vs_reference(lam, lam * 0 - 13.0, lam, lam * 0 - 10.0, 1550.0) - 3.0) < 1e-9
