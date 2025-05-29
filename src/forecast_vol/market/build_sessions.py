"""
Build NYSE active-session calendar.

Details
-------
For every calendar day between start_date and end_date (inclusive),
this script computes the UTC epoch-millisecond open/close timestamps for
the regular NYSE session (09:30 to 16:00 America/New_York) or for the
shortened half-day session (09:30 to 13:00) defined in the exceptions
YAML. Results are written to `paths.active_sessions_json`.

Configuration (configs/market_sessions.yaml)
--------------------------------------------
paths.exceptions_yaml       : data/external/market_hours/exceptions.yaml
paths.active_sessions_json  : data/processed/nyse_2023-2024.json
paths.reports_dir           : reports/market_sessions
params.start_date           : "2023-01-04"
params.end_date             : "2024-12-31"

Outputs
-------
- `paths.active_sessions_json`  (main calendar artifacts)
- `paths.reports_dir`/session_calendar_report.json  (summary stats)

Public API
----------
- main : command-line entry-point used by `make build_sessions [--debug]`
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/market_sessions.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
_EXCEPTIONS_YAML = Path(_PATHS["exceptions_yaml"])
_ACTIVE_JSON = Path(_PATHS["active_sessions_json"])

REPORT_DIR = Path(_PATHS["reports_dir"])
ART_DIR = Path(_PATHS["artifacts_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)
ART_DIR.mkdir(parents=True, exist_ok=True)

START_DATE: date = date.fromisoformat(_CFG["params"]["start_date"])
END_DATE:   date = date.fromisoformat(_CFG["params"]["end_date"])

_TZ_NY = ZoneInfo("America/New_York")
_TZ_UTC = ZoneInfo("UTC")
OPEN_TIME = time(hour=9, minute=30)
CLOSE_FULL = time(hour=16, minute=0)
CLOSE_HALF = time(hour=13, minute=0)

# Logging
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def _read_exceptions(path: Path) -> tuple[set[date], set[date]]:
    """Load holiday and half-day dates from YAML into date sets."""
    raw = yaml.safe_load(path.read_text())

    def _as_dates(section: str) -> set[date]:
        out: set[date] = set()
        for lst in raw.get(section, {}).values():
            out.update(date.fromisoformat(d) for d in lst)
        return out

    holidays = _as_dates("holidays")
    half_days = _as_dates("half_days")
    overlap = holidays & half_days
    if overlap:
        msg = f"Date(s) appear in both holidays and half_days: {sorted(overlap)}"
        raise ValueError(msg)
    return holidays, half_days


def _daterange(start: date, end: date) -> Iterable[date]:
    """Yield dates from start to end inclusive."""
    cur = start
    step = timedelta(days=1)
    while cur <= end:
        yield cur
        cur += step


def _is_weekday(d: date) -> bool:
    # Monday == 0 ... Sunday == 6
    return d.weekday() < 5


def _to_epoch_ms(dt: datetime) -> int:
    """Convert timezone-aware datetime to epoch-ms int."""
    return int(dt.timestamp() * 1_000)

# --------------------------------------------------------------------------- #
# Core calendar construction
# --------------------------------------------------------------------------- #

def build_session_calendar(
    start: date,
    end: date,
    holidays: set[date],
    half_days: set[date],
) -> dict[str, list[int]]:
    """Return mapping day-str -> [open_ms, close_ms]."""
    calendar: dict[str, list[int]] = {}

    for day in _daterange(start, end):
        if not _is_weekday(day):
            continue
        if day in holidays:
            continue

        open_dt = datetime.combine(day, OPEN_TIME, tzinfo=_TZ_NY)
        close_time = CLOSE_HALF if day in half_days else CLOSE_FULL
        close_dt = datetime.combine(day, close_time, tzinfo=_TZ_NY)

        calendar[day.isoformat()] = [
            _to_epoch_ms(open_dt.astimezone(_TZ_UTC)),
            _to_epoch_ms(close_dt.astimezone(_TZ_UTC)),
        ]

    return calendar

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """Generate the NYSE session calendar."""
    parser = argparse.ArgumentParser("Build NYSE session calendar")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress messages",
    )
    args = parser.parse_args(argv)

    if args.debug:
        log.setLevel(logging.DEBUG)

    holidays, half_days = _read_exceptions(_EXCEPTIONS_YAML)
    cal = build_session_calendar(START_DATE, END_DATE, holidays, half_days)

    _ACTIVE_JSON.parent.mkdir(parents=True, exist_ok=True)
    _ACTIVE_JSON.write_text(json.dumps(cal, indent=2))

    # Summary report
    report = {
        "start_date": START_DATE.isoformat(),
        "end_date": END_DATE.isoformat(),
        "trading_days": len(cal),
        "full_days": sum(
            1 for d in cal if date.fromisoformat(d) not in half_days
        ),
        "half_days": len(half_days & {date.fromisoformat(d) for d in cal}),
        "holidays": len(holidays),
        "skipped_weekends": (
            (END_DATE - START_DATE).days + 1
            - len(cal)
            - len(holidays)
        ),
    }
    (REPORT_DIR / "session_calendar_report.json").write_text(
        json.dumps(report, indent=2)
    )

    log.info(
        "Built NYSE session calendar: %s trading days (%s full, %s half). "
        "Output -> %s",
        len(cal),
        report["full_days"],
        report["half_days"],
        _ACTIVE_JSON,
    )


if __name__ == "__main__":
    main()

