"""PyVISA drivers. One class per instrument, SCPI strings in one place.

The same classes open a virtual resource (TCPIP::127.0.0.1::5025::SOCKET) or a
real one (GPIB0::20::INSTR, TCPIP::192.168.1.10::inst0::INSTR); only the
resource string in the station config changes.
"""
from __future__ import annotations

import pyvisa


class ScpiInstrument:
    def __init__(self, resource: str, rm: pyvisa.ResourceManager | None = None, timeout_ms: int = 5000):
        self.resource = resource
        self.rm = rm or pyvisa.ResourceManager("@py")
        self.inst = self.rm.open_resource(
            resource, read_termination="\n", write_termination="\n", timeout=timeout_ms
        )

    def write(self, cmd: str) -> None:
        """Send a command and wait until the instrument has carried it out.

        The *OPC? query does two jobs. It blocks until the command is done, so a
        reading taken next on ANOTHER instrument (the power meter after a laser
        step) cannot overtake it. And it makes a rejected command visible: an
        instrument that answers the write with an error string would otherwise
        leave that string in the buffer for the next, unrelated query.
        """
        reply = self.inst.query(cmd + ";*OPC?").strip()
        if reply != "1":
            raise RuntimeError(f"{self.resource}: {cmd!r} answered {reply!r}")

    def query(self, cmd: str) -> str:
        return self.inst.query(cmd).strip()

    def query_float(self, cmd: str) -> float:
        return float(self.query(cmd))

    def idn(self) -> str:
        return self.query("*IDN?")

    def close(self) -> None:
        self.inst.close()


class TunableLaser(ScpiInstrument):
    def set_wavelength(self, nm: float) -> None:
        self.write(f":SOUR:WAV {nm:.4f}NM")

    def wavelength(self) -> float:
        return self.query_float(":SOUR:WAV?")

    def set_power_dbm(self, dbm: float) -> None:
        self.write(f":SOUR:POW {dbm:.3f}DBM")

    def output(self, on: bool) -> None:
        self.write(f":OUTP {1 if on else 0}")


class PowerMeter(ScpiInstrument):
    def set_cal_wavelength(self, nm: float) -> None:
        self.write(f":SENS:POW:WAV {nm:.3f}NM")

    def read_dbm(self, channel: int) -> float:
        return self.query_float(f":SENS{channel}:POW?")

    def read_all_dbm(self, channels) -> list[float]:
        return [self.read_dbm(c) for c in channels]


class SMU(ScpiInstrument):
    def set_compliance(self, channel: int, volts: float) -> None:
        self.write(f":SOUR{channel}:VOLT:LIM {volts:.3f}")

    def set_current(self, channel: int, amps: float) -> None:
        self.write(f":SOUR{channel}:CURR {amps:.6e}")

    def output(self, channel: int, on: bool) -> None:
        self.write(f":OUTP{channel} {1 if on else 0}")

    def measure_voltage(self, channel: int) -> float:
        return self.query_float(f":MEAS{channel}:VOLT?")

    def measure_current(self, channel: int) -> float:
        return self.query_float(f":MEAS{channel}:CURR?")


class Prober:
    """Prober / chuck. Real probers have their own protocols; the virtual one speaks SCPI-like text."""

    def __init__(self, resource: str, rm: pyvisa.ResourceManager | None = None):
        self._i = ScpiInstrument(resource, rm)

    def idn(self) -> str:
        return self._i.idn()

    def load_die(self, wafer: str, die: str) -> None:
        self._i.write(f":MOVE:DIE {wafer},{die}")

    def select_bank(self, bank: int) -> None:
        self._i.write(f":MOVE:BANK {bank}")

    def align(self) -> None:
        self._i.write(":ALIGN")

    def contact(self) -> bool:
        return self._i.query(":CONT?") == "1"

    def close(self) -> None:
        self._i.close()
