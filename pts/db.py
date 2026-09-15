"""Measurement database: SQLite, four tables, pandas in and out.

runs      one row per measurement routine on one die (who, when, what, parameters)
traces    the raw sweep rows of a run (long format: run, channel, x, y)
results   the numbers analysis extracted (run, metric, value)
dies      one row per die with its latest status
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
  run_id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, station TEXT, operator TEXT,
  wafer TEXT NOT NULL, die TEXT NOT NULL, bank INTEGER,
  kind TEXT NOT NULL, params TEXT
);
CREATE TABLE IF NOT EXISTS traces (
  run_id INTEGER NOT NULL, channel TEXT NOT NULL,
  x REAL NOT NULL, y REAL NOT NULL, aux REAL,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS results (
  run_id INTEGER NOT NULL, metric TEXT NOT NULL, value REAL, unit TEXT,
  FOREIGN KEY(run_id) REFERENCES runs(run_id)
);
CREATE TABLE IF NOT EXISTS dies (
  wafer TEXT NOT NULL, die TEXT NOT NULL, status TEXT, note TEXT, ts TEXT,
  PRIMARY KEY (wafer, die)
);
CREATE INDEX IF NOT EXISTS ix_traces_run ON traces(run_id);
CREATE INDEX IF NOT EXISTS ix_results_metric ON results(metric);
"""

UNITS = {"nm": "nm", "db": "dB", "dbm": "dBm", "ohm": "ohm", "mw": "mW", "found": "bool", "notches": "count"}


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.executescript(SCHEMA)
    return con


def new_run(con, station: str, operator: str, wafer: str, die: str, bank: int, kind: str, params: dict) -> int:
    cur = con.execute(
        "INSERT INTO runs(ts, station, operator, wafer, die, bank, kind, params) VALUES (?,?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(timespec="seconds"), station, operator, wafer, die, bank, kind, json.dumps(params)),
    )
    con.commit()
    return int(cur.lastrowid)


def add_trace(con, run_id: int, df: pd.DataFrame, x_col: str = "wavelength_nm", aux_col: str | None = None) -> None:
    ycols = [c for c in df.columns if c not in (x_col, aux_col, "current_a", "voltage_v", "power_mw")]
    rows = []
    for c in ycols:
        aux = df[aux_col].values if aux_col else [None] * len(df)
        rows += [(run_id, c, float(x), float(y), (None if a is None else float(a))) for x, y, a in zip(df[x_col].values, df[c].values, aux)]
    con.executemany("INSERT INTO traces(run_id, channel, x, y, aux) VALUES (?,?,?,?,?)", rows)
    con.commit()


def add_results(con, run_id: int, metrics: dict) -> None:
    rows = []
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            unit = next((u for s, u in UNITS.items() if k.endswith("_" + s) or k.endswith(s)), "")
            rows.append((run_id, k, float(v), unit))
    con.executemany("INSERT INTO results(run_id, metric, value, unit) VALUES (?,?,?,?)", rows)
    con.commit()


def set_die_status(con, wafer: str, die: str, status: str, note: str = "") -> None:
    con.execute("INSERT OR REPLACE INTO dies(wafer, die, status, note, ts) VALUES (?,?,?,?,?)",
                (wafer, die, status, note, datetime.now(timezone.utc).isoformat(timespec="seconds")))
    con.commit()


def results(con, metric: str | None = None, kind: str | None = None) -> pd.DataFrame:
    q = ("SELECT r.run_id, r.ts, r.wafer, r.die, r.bank, r.kind, s.metric, s.value, s.unit "
         "FROM results s JOIN runs r ON r.run_id = s.run_id")
    conds, args = [], []
    if metric:
        conds.append("s.metric = ?"); args.append(metric)
    if kind:
        conds.append("r.kind = ?"); args.append(kind)
    if conds:
        q += " WHERE " + " AND ".join(conds)
    return pd.read_sql_query(q + " ORDER BY r.run_id", con, params=args)


def results_wide(con, kind: str = "wavelength_sweep") -> pd.DataFrame:
    """One row per run, one column per metric (the SPC input)."""
    r = results(con, kind=kind)
    if r.empty:
        return r
    return r.pivot_table(index=["run_id", "ts", "wafer", "die", "bank"], columns="metric", values="value").reset_index()


def trace(con, run_id: int) -> pd.DataFrame:
    t = pd.read_sql_query("SELECT channel, x, y, aux FROM traces WHERE run_id = ? ORDER BY channel, x", con, params=(run_id,))
    return t.pivot_table(index="x", columns="channel", values="y").reset_index().rename(columns={"x": "wavelength_nm"})


def dies(con) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM dies ORDER BY wafer, die", con)
