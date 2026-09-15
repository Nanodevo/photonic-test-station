"""Virtual instrument rack: SCPI over TCP sockets.

Four servers, one per instrument, share one DUT. The station code opens them
as VISA resources (TCPIP::127.0.0.1::<port>::SOCKET), exactly as it would open
a real laser or power meter, and speaks SCPI to them. The command set follows
the shape of Keysight 816x lasers / N77xx power meters and Keithley SMUs,
reduced to what the station needs.
"""
from __future__ import annotations

import re
import socketserver
import threading

from .dut import DUT

PORTS = {"laser": 5025, "power_meter": 5026, "smu": 5027, "prober": 5028}


def _on(v: str) -> bool:
    return v.strip().upper() in ("1", "ON")


class Rack:
    """Shared state behind all four instruments."""

    def __init__(self):
        self.lock = threading.RLock()
        self.dut = DUT()
        self.laser_nm = 1550.0
        self.laser_dbm = 0.0
        self.laser_on = False
        self.pm_cal_nm = 1550.0
        self.smu_on = [False] * 4

    # ---------------------------------------------------------- laser
    def laser(self, cmd: str) -> str | None:
        c = cmd.upper()
        if c == "*IDN?":
            return "PTS,VirtualTunableLaser,0001,0.1"
        if c == "*RST":
            self.laser_on, self.laser_nm, self.laser_dbm = False, 1550.0, 0.0
            return None
        if c == "*OPC?":
            return "1"
        m = re.fullmatch(r":?(?:SOUR(?:CE)?:)?WAV(?:ELENGTH)?\s+([\d.]+)\s*(NM)?", c)
        if m:
            self.laser_nm = float(m.group(1))
            return None
        if re.fullmatch(r":?(?:SOUR(?:CE)?:)?WAV(?:ELENGTH)?\?", c):
            return f"{self.laser_nm:.4f}"
        m = re.fullmatch(r":?(?:SOUR(?:CE)?:)?POW(?:ER)?\s+([-+\d.eE]+)\s*(DBM)?", c)
        if m:
            self.laser_dbm = float(m.group(1))
            return None
        if re.fullmatch(r":?(?:SOUR(?:CE)?:)?POW(?:ER)?\?", c):
            return f"{self.laser_dbm:.3f}"
        m = re.fullmatch(r":?OUTP(?:UT)?\s+(\w+)", c)
        if m:
            self.laser_on = _on(m.group(1))
            return None
        if re.fullmatch(r":?OUTP(?:UT)?\?", c):
            return "1" if self.laser_on else "0"
        return "ERR:UNKNOWN"

    # ---------------------------------------------------------- power meter
    def power_meter(self, cmd: str) -> str | None:
        c = cmd.upper()
        if c == "*IDN?":
            return "PTS,VirtualPowerMeter6ch,0002,0.1"
        if c in ("*RST", "*CLS"):
            return None
        if c == "*OPC?":
            return "1"
        m = re.fullmatch(r":?(?:SENS(?:E)?(\d)|READ(\d)):?(?:POW(?:ER)?)?\?", c)
        if m:
            ch = int(m.group(1) or m.group(2))
            if ch not in range(1, 7):
                return "ERR:CHANNEL"
            return f"{self.dut.power_dbm(ch, self.laser_nm, self.laser_dbm, self.laser_on):.3f}"
        if re.fullmatch(r":?SENS(?:E)?\d?:?POW(?:ER)?:UNIT\?", c):
            return "DBM"
        m = re.fullmatch(r":?SENS(?:E)?\d?:?POW(?:ER)?:WAV(?:ELENGTH)?\s+([\d.]+)\s*(NM)?", c)
        if m:
            self.pm_cal_nm = float(m.group(1))
            return None
        if re.fullmatch(r":?SENS(?:E)?\d?:?POW(?:ER)?:WAV(?:ELENGTH)?\?", c):
            return f"{self.pm_cal_nm:.3f}"
        return "ERR:UNKNOWN"

    # ---------------------------------------------------------- SMU (4 ch)
    def smu(self, cmd: str) -> str | None:
        c = cmd.upper()
        if c == "*IDN?":
            return "PTS,VirtualSMU4ch,0003,0.1"
        if c == "*RST":
            self.dut.heater_ma[:] = 0.0
            self.smu_on = [False] * 4
            return None
        if c == "*OPC?":
            return "1"
        m = re.fullmatch(r":?SOUR(?:CE)?(\d):?CURR(?:ENT)?\s+([-+\d.eE]+)\s*(A)?", c)
        if m:
            ch = int(m.group(1))
            self.dut.heater_ma[ch - 1] = float(m.group(2)) * 1e3
            return None
        m = re.fullmatch(r":?SOUR(?:CE)?(\d):?CURR(?:ENT)?\?", c)
        if m:
            return f"{self.dut.heater_ma[int(m.group(1)) - 1] * 1e-3:.6e}"
        m = re.fullmatch(r":?SOUR(?:CE)?(\d):?VOLT(?:AGE)?:LIM(?:IT)?\s+([\d.]+)", c)
        if m:
            self.dut.compliance_v = float(m.group(2))
            return None
        m = re.fullmatch(r":?MEAS(?:URE)?(\d):?VOLT(?:AGE)?\?", c)
        if m:
            ch = int(m.group(1))
            return f"{(self.dut.heater_voltage(ch) if self.smu_on[ch - 1] else 0.0):.5f}"
        m = re.fullmatch(r":?MEAS(?:URE)?(\d):?CURR(?:ENT)?\?", c)
        if m:
            ch = int(m.group(1))
            return f"{(self.dut.heater_current_actual(ch) if self.smu_on[ch - 1] else 0.0):.6e}"
        m = re.fullmatch(r":?OUTP(?:UT)?(\d)\s+(\w+)", c)
        if m:
            self.smu_on[int(m.group(1)) - 1] = _on(m.group(2))
            return None
        m = re.fullmatch(r":?OUTP(?:UT)?(\d)\?", c)
        if m:
            return "1" if self.smu_on[int(m.group(1)) - 1] else "0"
        return "ERR:UNKNOWN"

    # ---------------------------------------------------------- prober
    def prober(self, cmd: str) -> str | None:
        c = cmd.strip()
        u = c.upper()
        if u == "*IDN?":
            return "PTS,VirtualProber,0004,0.1"
        m = re.fullmatch(r":?MOVE:DIE\s+([\w-]+)\s*,\s*([\w-]+)", c, re.I)
        if m:
            self.dut.load(m.group(1), m.group(2))
            self.dut.heater_ma[:] = 0.0
            return None
        if u == ":DIE?" or u == "DIE?":
            p = self.dut.params
            return f"{p.wafer},{p.die}" if p else "NONE,NONE"
        m = re.fullmatch(r":?MOVE:BANK\s+([123])", u)
        if m:
            self.dut.bank = int(m.group(1))
            self.dut.heater_ma[:] = 0.0
            return None
        if u in (":BANK?", "BANK?"):
            return str(self.dut.bank)
        if u in (":ALIGN", "ALIGN"):
            self.dut.realign()
            return None
        if u in (":CONT?", ":CONTACT?", "CONT?"):
            return "1" if self.dut.params else "0"
        return "ERR:UNKNOWN"


def _make_handler(rack: Rack, fn):
    class Handler(socketserver.StreamRequestHandler):
        def handle(self):
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                for cmd in line.decode(errors="ignore").strip().split(";"):
                    cmd = cmd.strip()
                    if not cmd:
                        continue
                    with rack.lock:
                        reply = fn(cmd)
                    if reply is not None:
                        self.wfile.write((reply + "\n").encode())
    return Handler


class _Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True


def start(rack: Rack | None = None, host: str = "127.0.0.1", ports: dict | None = None) -> tuple[Rack, dict]:
    """Start the four instrument servers in daemon threads. Returns (rack, servers)."""
    rack = rack or Rack()
    ports = ports or PORTS
    fns = {"laser": rack.laser, "power_meter": rack.power_meter, "smu": rack.smu, "prober": rack.prober}
    servers = {}
    for name, fn in fns.items():
        srv = _Server((host, ports[name]), _make_handler(rack, fn))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers[name] = srv
    return rack, servers


def resources(host: str = "127.0.0.1", ports: dict | None = None) -> dict:
    ports = ports or PORTS
    return {k: f"TCPIP::{host}::{v}::SOCKET" for k, v in ports.items()}
