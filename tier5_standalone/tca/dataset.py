"""One place that turns CLI arguments into a prepared frame.

Every tier's `run.py` calls this, so all three always score the identical set of
rows. That is what makes the head-to-head comparison in the top-level `run.py`
an apples-to-apples one.
"""

from __future__ import annotations
import argparse
import os

import pandas as pd

import config
import synthetic_data
from tca import pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")


def add_common_args(ap: argparse.ArgumentParser) -> argparse.ArgumentParser:
    ap.add_argument("--csv", help="Path to a real extract; omit to use synthetic data.")
    ap.add_argument("--n", type=int, default=12000, help="Synthetic row count.")
    ap.add_argument("--seed", type=int, default=7, help="Synthetic seed.")
    return ap


def load_prepared(args, quiet: bool = False):
    """Read (or generate) the book and run it through the shared pipeline."""
    if getattr(args, "csv", None):
        if not quiet:
            print(f"Loading extract: {args.csv}")
        df_raw = pd.read_csv(args.csv)
    else:
        if not quiet:
            print(f"No --csv given; generating {args.n:,} synthetic HK VWAP orders.")
        df_raw = synthetic_data.generate(n=args.n, seed=args.seed)

    df, clean_report = pipeline.prepare(
        df_raw, config.COLUMN_MAP, config.DATA, config.SLIPPAGE_SIGN,
        pre_transform=getattr(config, "PRE_TRANSFORM", None))
    return df, clean_report


def out_path(*parts: str) -> str:
    """Absolute path inside outputs/, creating directories as needed."""
    path = os.path.join(OUT_DIR, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
