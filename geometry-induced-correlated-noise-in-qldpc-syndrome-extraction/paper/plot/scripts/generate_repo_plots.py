#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate the full repository plot suite from a CSV file.

Example
-------
python scripts/generate_repo_plots.py --csv data/results.csv --outdir output/repo_plots
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from bbstim.plotting import plot_v3_suite

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=Path, default=Path('data/results.csv'))
    ap.add_argument('--outdir', type=Path, default=Path('output/repo_plots'))
    args = ap.parse_args()
    df = pd.read_csv(args.csv)
    plot_v3_suite(df, args.outdir)

if __name__ == '__main__':
    main()
