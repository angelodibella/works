#!/usr/bin/env python3
# Copyright (c) 2024-2026 Angelo Di Bella. MIT License.
"""Generate the publication figure suite and numeric macros.

This is a thin wrapper around scripts_generate_publication_assets.py using the
CSV inputs bundled in ./data.
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', type=Path, default=Path('output/publication_figures'))
    ap.add_argument('--numbers-tex', type=Path, default=Path('output/generated/numbers.tex'))
    args = ap.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env['PYTHONPATH'] = str(repo_root) + (os.pathsep + env['PYTHONPATH'] if 'PYTHONPATH' in env else '')
    cmd = [sys.executable, str(repo_root / 'scripts_generate_publication_assets.py'),
           '--repo-root', str(repo_root), '--outdir', str(args.outdir), '--numbers-tex', str(args.numbers_tex)]
    raise SystemExit(subprocess.call(cmd, env=env))

if __name__ == '__main__':
    main()
