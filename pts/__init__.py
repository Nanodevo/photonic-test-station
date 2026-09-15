"""pts: a photonic test station.

Station code talks SCPI over VISA (PyVISA). In virtual mode the instruments are small
TCP servers (pts.virtual_rack) wrapping a physics model of the WDM4 ring bank
(pts.dut); in real mode the same code opens real resources. Nothing in station.py,
sweeps or analysis knows which one it is talking to.
"""
__version__ = "0.1.0"
