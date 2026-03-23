# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
# See https://github.com/angelodibella/works for details.
import argparse
from pathlib import Path

import pandas as pd

from .experiments import DEFAULT_NUM_WORKERS, SUITES, clopper_pearson, derived_per_cycle_rate, run_suite
from .plotting import plot_v3_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BB Stim experiment utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run-suite", help="Run a named experiment suite."
    )
    run_parser.add_argument("--suite", required=True, choices=sorted(SUITES.keys()))
    run_parser.add_argument("--out", default="results/results.csv")
    run_parser.add_argument("--append", action="store_true")
    run_parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    run_parser.add_argument("--no-progress", action="store_true")
    run_parser.add_argument("--slice", default=None, help="Python-style slice of experiments, e.g. '0:4' or '4:8'")
    run_parser.add_argument("--shots-override", type=int, default=None, help="Override the shot count for all experiments in the suite.")

    merge_parser = subparsers.add_parser(
        "merge-csv", help="Merge split-run CSVs, summing shots/failures and recomputing CIs."
    )
    merge_parser.add_argument("csvs", nargs="+", help="CSV files to merge.")
    merge_parser.add_argument("--out", required=True, help="Output merged CSV path.")

    plot_parser = subparsers.add_parser(
        "plot-suite", help="Generate plots from a CSV of suite results."
    )
    plot_parser.add_argument("--csv", required=True)
    plot_parser.add_argument("--outdir", default="results/plots")
    return parser


def run_suite_command(*, suite: str, out: str, append: bool, num_workers: int, show_progress: bool, experiment_slice: str | None = None, shots_override: int | None = None) -> int:
    df = run_suite(suite, num_workers=num_workers, show_progress=show_progress, experiment_slice=experiment_slice, shots_override=shots_override)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.exists():
        old = pd.read_csv(out_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(df.to_string(index=False))
    print(f"Wrote {len(df)} rows to {out_path}")
    return 0


def merge_csv_command(*, csvs: list[str], out: str) -> int:
    """Merge split-run CSVs by summing failures/shots and recomputing CIs."""
    import numpy as np

    frames = [pd.read_csv(c) for c in csvs]
    combined = pd.concat(frames, ignore_index=True)

    # Columns that identify the same experiment (everything except results)
    id_cols = [c for c in combined.columns if not any(
        c.startswith(pfx) for pfx in ('primary_failures', 'primary_shots', 'primary_discards',
                                       'primary_ler', 'primary_decoder_seconds', 'primary_cpu',
                                       'secondary_failures', 'secondary_shots', 'secondary_discards',
                                       'secondary_ler', 'secondary_cpu', 'secondary_note',
                                       'wall_seconds')
    )]

    # For rows with a unique experiment_id, no merging needed
    groups = combined.groupby('experiment_id')
    rows = []
    for exp_id, grp in groups:
        if len(grp) == 1:
            rows.append(grp.iloc[0].to_dict())
            continue

        # Merge: take id columns from first row, sum count columns
        base = grp.iloc[0][id_cols].to_dict()
        pf = int(grp['primary_failures'].sum())
        ps = int(grp['primary_shots'].sum())
        pd_ = int(grp['primary_discards'].sum()) if 'primary_discards' in grp else 0
        cycles = int(base.get('cycles', 1))

        if ps > 0:
            p = pf / ps
            lo, hi = clopper_pearson(pf, ps)
            base.update({
                'primary_failures': pf,
                'primary_shots': ps,
                'primary_discards': pd_,
                'primary_ler_total': p,
                'primary_ler_total_lo': lo,
                'primary_ler_total_hi': hi,
                'primary_ler_per_cycle_derived': derived_per_cycle_rate(p, cycles),
                'primary_decoder_seconds': grp['primary_decoder_seconds'].sum() if 'primary_decoder_seconds' in grp else 0.0,
                'primary_cpu_seconds': grp['primary_cpu_seconds'].sum() if 'primary_cpu_seconds' in grp else 0.0,
            })
        else:
            base.update({
                'primary_failures': 0, 'primary_shots': 0, 'primary_discards': 0,
                'primary_ler_total': np.nan,
            })

        # Secondary decoder (if present)
        if 'secondary_shots' in grp.columns and grp['secondary_shots'].notna().any():
            sf = int(grp['secondary_failures'].sum())
            ss = int(grp['secondary_shots'].sum())
            if ss > 0:
                p2 = sf / ss
                lo2, hi2 = clopper_pearson(sf, ss)
                base.update({
                    'secondary_failures': sf,
                    'secondary_shots': ss,
                    'secondary_ler_total': p2,
                    'secondary_ler_total_lo': lo2,
                    'secondary_ler_total_hi': hi2,
                    'secondary_ler_per_cycle_derived': derived_per_cycle_rate(p2, cycles),
                })

        base['wall_seconds'] = grp['wall_seconds'].sum() if 'wall_seconds' in grp else 0.0
        rows.append(base)

    merged = pd.DataFrame(rows)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"Merged {len(frames)} CSVs → {len(merged)} experiments → {out_path}")
    return 0


def plot_suite_command(*, csv: str, outdir: str) -> int:
    df = pd.read_csv(csv)
    output_dir = Path(outdir)
    plot_v3_suite(df, output_dir)
    print(f"Plots written to {output_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run-suite":
        return run_suite_command(
            suite=args.suite,
            out=args.out,
            append=args.append,
            num_workers=args.num_workers,
            show_progress=not args.no_progress,
            experiment_slice=args.slice,
            shots_override=args.shots_override,
        )
    if args.command == "merge-csv":
        return merge_csv_command(csvs=args.csvs, out=args.out)
    return plot_suite_command(csv=args.csv, outdir=args.outdir)


if __name__ == "__main__":
    raise SystemExit(main())
