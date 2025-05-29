"""
Trim raw 1-minute CSV bars to the NYSE trading grid and audit gaps.

Details
--------
Crawls `paths.raw_root` for files named `<TICKER>_1m.csv` (spread
across year / asset-class sub-folders), merges them, keeps only rows whose
epoch-ms timestamps lie inside the UTC open/close span provided by
`paths.calendar_json`, writes the restricted parquet, and records per-ticker
gap statistics.

Configuration (configs/preprocess.yaml)
---------------------------------------
paths.raw_root        : data/raw
paths.calendar_json   : data/processed/nyse_2023-2024.json
paths.restricted_out  : data/interim/restricted
paths.reports_dir     : reports/preprocess

Outputs
-------
- `paths.restricted_out`/<TICKER>.parquet
- `paths.reports_dir`/restrict_gap_report.csv

Public API
----------
- main : command-line entry-point used by `make restrict [--debug]`
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import polars as pl
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/preprocess.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
ONE_MIN_MS: int = 60_000

CSV_READ_KW: dict[str, Any] = {
    "schema_overrides": {
        "volume": pl.Float32,
        "vwap": pl.Float32,
        "open": pl.Float32,
        "close": pl.Float32,
        "high": pl.Float32,
        "low": pl.Float32,
        "t": pl.Int64,
        "trades_count": pl.Int32,
    }
}

# Logging 
log = logging.getLogger(__name__.split(".")[-1])
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    log.addHandler(_handler)
log.setLevel(logging.INFO)


# --------------------------------------------------------------------------- #
# Report dataclass
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class GapStats:
    """Single ticker summary statistics after restriction.

    Attributes
    ----------
    ticker
        Upper-case ticker symbol.
    rows_before
        Row count prior to trimming.
    rows_after
        Row count after trimming.
    traded_all_sessions
        True if the ticker traded on every calendar session.
    missing_ticks
        Total number of missing minute bars.
    missing_blocks
        Count of contiguous gaps.
    blocks_at_open
        How many gap blocks include the session open.
    blocks_at_close
        How many gap blocks include the session close.
    nan_cells
        Total number of NaN cells.
    nan_cols
        Number of columns containing at least one NaN.
    """

    ticker: str
    rows_before: int
    rows_after: int
    traded_all_sessions: bool
    missing_ticks: int
    missing_blocks: int
    blocks_at_open: int
    blocks_at_close: int
    nan_cells: int
    nan_cols: int

    def to_dict(self) -> dict[str, Any]:
        """Return the dataclass as a plain dict with bool converted to int."""
        d = asdict(self)
        d["traded_all_sessions"] = int(self.traded_all_sessions)
        return d

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #

def load_sessions(path: Path) -> pl.DataFrame:
    """Load the trading calendar JSON file into a Polars DataFrame."""
    raw = json.loads(path.read_text())
    return pl.DataFrame(
        {
            "date": list(raw.keys()),
            "open_ms": [v[0] for v in raw.values()],
            "close_ms": [v[1] for v in raw.values()],
        }
    )


def discover_files(root: Path) -> dict[str, list[Path]]:
    """Recursively locate all files with suffix: "*_1m.csv" and group by ticker.

    Parameters
    ----------
    root
        Directory that contains yearly sub-folders with etf/ and stock/
        directories.

    Returns
    -------
    dict[str, list[pathlib.Path]]
        Mapping: ticker -> list_of_csv_files
    """
    tickers: dict[str, list[Path]] = defaultdict(list)
    for p in root.rglob("*_1m.csv"):
        tickers[p.stem.split("_")[0].upper()].append(p)
    return tickers


def contiguous_blocks(series: pl.Series) -> int:
    """Count contiguous blocks in a sorted series of missing timestamps."""
    if series.len() == 0:
        return 0
    diffs = series.sort().diff()
    return int((diffs != ONE_MIN_MS).sum() + 1)

# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #

def restrict_single(
    files: list[Path],
    sessions: pl.DataFrame
    ) -> tuple[pl.DataFrame, GapStats]:
    """Restrict one ticker to the NYSE grid and produce gap statistics.

    Parameters
    ----------
    files
        CSV files for a single ticker, possibly spanning multiple years.
    sessions
        Trading calendar as returned by ``load_sessions``.

    Returns
    -------
    tuple
        (cleaned_frame, gap_stats)
    """
    df_raw = pl.concat([pl.read_csv(p, **CSV_READ_KW) for p in files])
    rows_before = df_raw.height

    # Convert epoch ms to UTC datetime and join the calendar
    df = (
        df_raw.with_columns(
            pl.col("t")
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("datetime")
        )
        .with_columns(pl.col("datetime").dt.strftime("%Y-%m-%d").alias("date"))
        .join(sessions, on="date", how="left")
        .filter(
            (pl.col("t") >= pl.col("open_ms"))
            & (pl.col("t") <= pl.col("close_ms") - ONE_MIN_MS)
        )
        .drop(["open_ms", "close_ms"])
        .rechunk()
    )

    # Gap and NaN statistics
    present_dates = set(df.get_column("date"))
    missing_ticks = missing_blocks = blocks_open = blocks_close = 0

    for r in sessions.iter_rows(named=True):
        d, o, c = r["date"], r["open_ms"], r["close_ms"]
        expected = pl.Series(range(o, c, ONE_MIN_MS), dtype=pl.Int64)
        if d in present_dates:
            got = df.filter(pl.col("date") == d).get_column("t")
            miss = expected.filter(~expected.is_in(got))
            if miss.len():
                missing_ticks += miss.len()
                missing_blocks += contiguous_blocks(miss)
                if expected[0] in miss:
                    blocks_open += 1
                if expected[-1] in miss:
                    blocks_close += 1
        else:
            missing_ticks += expected.len()
            missing_blocks += 1
            blocks_open += 1
            blocks_close += 1

    na_counts = df.null_count().to_series()
    stats = GapStats(
        ticker=files[0].stem.split("_")[0].upper(),
        rows_before=rows_before,
        rows_after=df.height,
        traded_all_sessions=(len(present_dates) == sessions.height),
        missing_ticks=missing_ticks,
        missing_blocks=missing_blocks,
        blocks_at_open=blocks_open,
        blocks_at_close=blocks_close,
        nan_cells=int(na_counts.sum()),
        nan_cols=int((na_counts > 0).sum()),
    )

    return df, stats

# --------------------------------------------------------------------------- #
# Driver helpers
# --------------------------------------------------------------------------- #

def restrict_all(debug: bool = False):
    """Process every ticker found under paths.raw_root.

    Parameters
    ----------
    debug
        If True, emit DEBUG logs and show a progress bar.

    Returns
    -------
    tuple
        (ticker_frames, report_dataframe)
    """
    raw_root = Path(_PATHS["raw_root"])
    sessions = load_sessions(Path(_PATHS["calendar_json"]))
    ticker_map = discover_files(raw_root)
    log.info("Found %d tickers", len(ticker_map))

    frames: dict[str, pl.DataFrame] = {}
    report_rows: list[GapStats] = []

    bar = None
    if debug:
        bar = tqdm(
            total=len(ticker_map),
            desc="Restricting",
            unit="ticker",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )

    for ticker, files in ticker_map.items():
        df_tkr, stats = restrict_single(files, sessions)
        frames[ticker] = df_tkr
        report_rows.append(stats)
        log.debug(
            "%s done: %d -> %d rows",
            ticker,
            stats.rows_before,
            stats.rows_after
            )
        if bar:
            bar.update(1)

    if bar:
        bar.close()

    report_df = pl.DataFrame([r.to_dict() for r in report_rows]).to_pandas()
    return frames, report_df

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for minute restricting raw data to the trading calendar."""
    parser = argparse.ArgumentParser(
        "Restrict 1-minute bars to NYSE grid and audit gaps"
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress"
        )
    args = parser.parse_args(argv)

    if args.debug:
        log.setLevel(logging.DEBUG)
        
    frames, report = restrict_all(debug=args.debug)

    out_dir = Path(_PATHS["restricted_out"])
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for tkr, df in frames.items():
        df.write_parquet(out_dir / f"{tkr}.parquet")
        
    log.info("Wrote %d parquet files -> %s", len(frames), out_dir)

    report_path = Path(_PATHS["reports_dir"]) / "restrict_gap_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)
    log.info("Wrote gap report      -> %s", report_path)


if __name__ == "__main__":
    main()
