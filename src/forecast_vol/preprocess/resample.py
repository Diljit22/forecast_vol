"""
Gap-fill restricted minute bars and recompute VWAP.

Details
--------
For each Parquet file under `paths.restricted_out` (output of
`make restrict`), this script fills numeric gaps, synthesises missing
High/Low using an `epsilon_percent` wiggle, recomputes row-level VWAP,
and writes a cleaned Parquet per ticker to `paths.resampled_out`.
A per-ticker gap report is also recorded.

Configuration (configs/preprocess.yaml)
---------------------------------------
paths.restricted_out  : data/interim/restricted
paths.resampled_out   : data/interim/resampled
paths.calendar_json   : data/processed/nyse_2023-2024.json
params.epsilon_percent: 0.002
paths.reports_dir     : reports/preprocess

Outputs
-------
- `paths.resampled_out`/<TICKER>.parquet
- `paths.reports_dir`/resample_gap_report.csv

Public API
----------
- main : command-line entry-point used by `make resample [--debug]`
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import yaml
from tqdm import tqdm

# --------------------------------------------------------------------------- #
# Configuration & global constants
# --------------------------------------------------------------------------- #

_CFG_PATH = Path("configs/preprocess.yaml")
_CFG = yaml.safe_load(_CFG_PATH.read_text())
_PATHS = _CFG["paths"]
EPS = float(_CFG["params"]["epsilon_percent"])
RNG = np.random.default_rng()
ONE_MIN_MS: int = 60_000

SRC_DIR = Path(_PATHS["restricted_out"])
DST_DIR = Path(_PATHS["resampled_out"])
REPORT_DIR = Path(_PATHS["reports_dir"])
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# NYSE calendar
_CAL_JS = Path(_PATHS["calendar_json"])
_raw_cal = json.loads(_CAL_JS.read_text())
CAL: dict[str, tuple[int, int]] = {
    d: (span[0], span[1]) for d, span in _raw_cal.items()
    }

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

def _minute_grid(days: list[str]) -> pl.Series:
    """Return the expected epoch-ms grid for a collection of trading days."""
    buf: list[int] = []
    for d in days:
        o, c = CAL[d]
        buf.extend(range(o, c, ONE_MIN_MS))
    return pl.Series(buf, dtype=pl.Int64)

def _interpolate_numeric(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Interpolate numeric cols where null."""
    out = df
    for col in cols:
        if col in out.columns:
            out = out.with_columns(pl.col(col).interpolate())
    return out

# --------------------------------------------------------------------------- #
# Core processing
# --------------------------------------------------------------------------- #

def process_ticker(path: Path) -> dict[str, object]:
    """Gap-fill and resample a single ticker's parquet file.

    Parameters
    ----------
    path
        Parquet file created by *restrict.py*.

    Returns
    -------
    dict
        Dictionary with basic statistics for the ticker.
    """
    tkr = path.stem.upper()
    df = pl.read_parquet(path)

    # Add UTC datetime and trading day string
    df = (
        df.sort("t")
        .with_columns(
            pl.col("t")
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("dt")
        )
        .with_columns(
            pl.col("dt")
            .dt.strftime("%Y-%m-%d")
            .alias("day")
            )
    )

    # Build full minute grid for the days present
    days = df.get_column("day").unique().to_list()
    full = pl.DataFrame({"t": _minute_grid(sorted(days))})

    open_mean_before = float(df["open"].mean())
    open_std_before = float(df["open"].std(ddof=0))

    merged = (
        full.join(df, on="t", how="left")
        .sort("t")
        .with_columns(
            pl.col("t")
            .cast(pl.Datetime("ms"))
            .dt.replace_time_zone("UTC")
            .alias("dt")
        )
    )

    before = df.height
    initial_gaps = merged.filter(pl.col("open").is_null()).height
    mask_gap = merged["open"].is_null()

    # Interpolate null numeric cols
    merged = _interpolate_numeric(merged, ["open", "close", "volume", "vwap"])

    # Synthetic High / Low wiggle for rows that were gaps
    if EPS > 0 and initial_gaps:
        rnd = RNG.uniform(0, EPS, size=merged.height)
        merged = merged.with_columns(pl.Series("_rnd", rnd))
        merged = merged.with_columns(
            [
                pl.when(mask_gap)
                .then(pl.col("close") * (1 + pl.col("_rnd")))
                .otherwise(
                    pl.when(pl.col("high").is_null())
                    .then(pl.max_horizontal("open", "close"))
                    .otherwise(pl.col("high"))
                )
                .alias("high"),
                pl.when(mask_gap)
                .then(pl.col("close") * (1 - pl.col("_rnd")))
                .otherwise(
                    pl.when(pl.col("low").is_null())
                    .then(pl.min_horizontal("open", "close"))
                    .otherwise(pl.col("low"))
                )
                .alias("low"),
            ]
        ).drop("_rnd")

    # Recompute VWAP 
    merged = merged.with_columns(
        ((pl.col("open") + pl.col("close")) / 2).alias("vwap")
        )

    # Final forward/backward fill to remove created nulls
    merged = merged.fill_null("forward").fill_null("backward")

    # Write outputs
    DST_DIR.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(DST_DIR / path.name)

    open_mean_after = float(merged["open"].mean())
    open_std_after = float(merged["open"].std(ddof=0))

    return {
        "ticker": tkr,
        "rows_before": before,
        "rows_after": merged.height,
        "gaps_filled": initial_gaps,
        "open_mean_before": open_mean_before,
        "open_std_before": open_std_before,
        "open_mean_after": open_mean_after,
        "open_std_after": open_std_after,
    }

# --------------------------------------------------------------------------- #
# CLI entrypoint
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> None:
    """CLI entrypoint for gap-filling and resampling."""
    parser = argparse.ArgumentParser(
        "Gap-fill and resample restricted parquet"
        )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="DEBUG logs + progress"
        )
    args = parser.parse_args(argv)
    
    if args.debug:
        log.setLevel(logging.DEBUG)

    files = sorted(SRC_DIR.glob("*.parquet"))
    if not files:
        log.error("No parquet in %s", SRC_DIR)
        return

    iterator = files
    if args.debug:
        iterator = tqdm(
            files,
            desc="Gap-filling",
            unit="ticker",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}",
        )

    rows_report: list[dict[str, object]] = []
    for fp in iterator:
        stats = process_ticker(fp)
        rows_report.append(stats)

        log.debug(
            f"{stats['ticker']}: gaps={stats['gaps_filled']} "
            f"open mu(before)={stats['open_mean_before']:.4f} "
            f"mu(after)={stats['open_mean_after']:.4f}"
        )
        if not args.debug and stats["gaps_filled"] == 0:
            log.info("%s - no gaps", stats["ticker"])

    pl.DataFrame(rows_report).write_csv(REPORT_DIR / "resample_gap_report.csv")
    log.info("Resampling complete. Output -> %s", DST_DIR)


if __name__ == "__main__":
    main()
