# Geometry-Induced Correlated Noise in QLDPC Syndrome Extraction

Simulation pipeline, processed data, and figure-generation scripts for the paper:

> A. Di Bella, "Geometry-induced correlated noise in qLDPC syndrome extraction" (2026).

If you use this code, please cite using the metadata in [`CITATION.cff`](CITATION.cff).

## Contents

- **`bbstim/`** — Python package for BB (bivariate bicycle) code simulation under geometry-induced correlated noise. Implements circuit-level syndrome extraction, multiple embedding strategies (single-layer monomial, biplanar bounded-thickness, logical-aware column-permutation), and the BP+OSD decoder via `stim` and `sinter`.
- **`scripts/`** — Deterministic audits: logical-family enumeration, geometry metrics, logical-aware optimisation, microscopic diagnostics, decoder-aware evaluation.
- **`tests/`** — Unit tests covering algebraic correctness, embedding geometry, and circuit construction.
- **`configs/`** — Frozen logical-aware embedding (BB72, 36-support pure-q(L) family, swap-descent optimised) and random-layout configurations.
- **`results/`** — Raw and semantically merged CSV data for BB72, BB90, BB108, BB144.
- **`results-out/`** — Additional results: C_D evaluation, many-layout validation, proxy-kernel sweep, robustness checks.
- **`paper/`** — Paper source (LaTeX), figure-generation scripts, TikZ sources, and processed data.

## Codes

| Code | Parameters | Lattice | Polynomials |
|------|-----------|---------|-------------|
| BB72 | [[72,12,6]] | Z6 x Z6 | A = x^3+y+y^2, B = y^3+x+x^2 |
| BB90 | [[90,8,10]] | Z15 x Z3 | A = x^9+y+y^2, B = 1+x^2+x^7 |
| BB108 | [[108,8,12]] | Z9 x Z6 | A = x^3+y+y^2, B = y^3+x+x^2 |
| BB144 | [[144,12,12]] | Z12 x Z6 | A = x^3+y+y^2, B = y^3+x+x^2 |

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
python -m pytest tests/
```

## Reproducing figures

```bash
cd paper/plot
python scripts_generate_publication_assets.py --repo-root ../.. --outdir ../figures --numbers-tex ../generated/numbers.tex
```

## Licence

MIT. See `LICENSE` in the repository root.
