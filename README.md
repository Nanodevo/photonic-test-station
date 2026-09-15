# pts: a photonic test station

Station software for die-level testing of photonic chips, built around a
**virtual instrument rack** so that the whole loop can run without a lab:

```
 scripts/measure_lot.py
        |
   pts.routines  ── load die, check heaters, sweep, analyse, judge, store
        |
   pts.station   ── wavelength_sweep(), heater_sweep(), heater_resistances()
        |
   pts.instruments ── TunableLaser / PowerMeter / SMU / Prober  (PyVISA, SCPI)
        |  VISA resource strings from config/*.json
        v
 TCPIP::127.0.0.1::5025..5028::SOCKET        GPIB0::20::INSTR, TCPIP::...::inst0::INSTR
   pts.virtual_rack (this repo)                 real instruments (same code, other config)
        |
   pts.dut ── physics of the WDM4 ring bank, per-die process variation, defects
```

The station code never knows whether it talks to the virtual rack or to a real
laser: it writes SCPI strings to a VISA resource and reads the reply. Swapping
in real hardware is a change of `config/virtual.json` to `config/real.example.json`.

## The chip

The device under test is the WDM4 die from
[cornerstone-pic](https://nanodevo.github.io/reports/wdm4-demux.html): four
add-drop racetrack rings on CORNERSTONE's 220 nm SOI platform, one TiN heater
per ring, laid out three times with coupling gaps of 0.22 / 0.27 / 0.32 µm,
grating couplers on the fiber-array pitch and a reference loopback. The model
in `pts/dut.py` uses the femwell-derived waveguide numbers of that design
(`data/strip_450x220.json`), adds process variation seeded per (wafer, die)
(width, thickness, gap, loss, coupler centre) and gives about 8 % of the dies a
defect: a particle on a ring, an open heater, a shorted heater.

## What the station measures

| routine | instruments | numbers extracted |
|---|---|---|
| `heater_resistances()` | SMU, 1 mA per heater | R per heater; open (compliance) / short |
| `wavelength_sweep()` | laser step, 6 power-meter channels | resonance per ring, FWHM, FSR, drop loss vs loopback, through-notch depth, channel spacing |
| `heater_sweep()` | SMU current steps + narrow sweeps | nm/mW tuning slope, P_π |

Every run goes into SQLite (`pts/db.py`: runs, traces, results, dies), each
die gets a PASS/FAIL against the limits in `pts/routines.py`, and
`scripts/spc_report.py` draws individuals charts (limits frozen on the
baseline, plasmafab's convention) with process capability per metric.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e . pytest "git+https://github.com/Nanodevo/plasma-process-toolkit"
python -m pytest -q

python scripts/rack.py                    # terminal 1: the virtual instruments
python scripts/measure_die.py --wafer W01 --die D07 --plot   # terminal 2
python scripts/measure_lot.py --wafers W01 W02 --dies 25
python scripts/spc_report.py
```

## What is and is not real

The instruments are simulated; the SCPI command set is modelled on Keysight
816x / N77xx and Keithley SMU syntax but reduced to what the station needs.
The physics is the analytic add-drop ring response with first-order
dispersion; there is no RF path yet. The point of the repository is the
station software: drivers, sweeps, analysis, database, acceptance limits and
control charts, written so that the day a real probe station is on the other
end of the cable, only the config changes.

## Layout

```
pts/            package: dut, virtual_rack, instruments, station, analysis, db, routines
scripts/        rack, measure_die, measure_lot, spc_report
config/         virtual.json (the rack), real.example.json (what a real station would look like)
data/           waveguide numbers from the design; results.sqlite is created here
docs/img/       figures written by the scripts
tests/          pytest: analysis on synthetic traces, DUT physics, defect behaviour
```
