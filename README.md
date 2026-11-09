# Ghost Data Engine 3.3

Auditable, pure-Python gap filling for gridded data (2D grids, CSV tables and space-time stacks).
It finds voids (connected missing regions), measures the tension of the recorded cells around each
void, and fills it with inward propagation waves, optional smoothing and optional post-fill blends.
**Every filled cell has an audit row**: its source cells, wave number and a heuristic confidence.
Filled values are estimates, not recovered truth.

Author: Maurice Gathungu. Licence: MIT.

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py          # web app (upload limit 500 MB, see below)
```

Command line (no extra dependencies):

```bash
python main.py --width 30 --height 30 --seed 7        # synthetic demo + report
python engine_v3.py                                    # repairs a tiny sample CSV
python gui_app.py                                      # Tkinter desktop console
```

Python API:

```python
from engine_v3 import process_and_fix_file
process_and_fix_file("input.csv", "output_repaired.csv")      # blends on by default
```

## Upload size

The web app accepts uploads up to **500 MB** (`.streamlit/config.toml`, `server.maxUploadSize = 500`).
Two practical limits still apply, because the engine is pure Python:

| Limit | Default | Override (environment variable) |
|---|---|---|
| Cells repaired per request, 2D grid | 60,000 | `GDE_MAX_CELLS_2D` |
| Cells repaired per request, table | 60,000 | `GDE_MAX_TABLE_CELLS` |
| Cells per 3D run | 12,000 | `GDE_MAX_CELLS_3D` |

Measured speed is about 1 s at 13k cells and 9 s at 50k cells on one core. Bigger files are accepted and
scanned, but over the cap the app asks you to use the command line. Free hosting (about 1 GB RAM) cannot hold
a 500 MB file in memory once it is parsed, so for files that large run the app on a larger machine and raise
the caps, for example `GDE_MAX_CELLS_2D=2000000 streamlit run app.py`.

## Layout

```
app.py, app_core.py            Streamlit web app and its Streamlit-free helpers
engine_v3.py                   CSV pipeline: load -> scan -> tension -> reconstruct -> blends -> export
scanner.py, tension.py,        The engine: void detection, boundary tension, wave reconstruction
reconstructor.py
gde_spatiotemporal.py          3D (time, y, x) engine with temporal and spatial sources
generator.py                   Synthetic grids with irregular voids (and ground truth for tests)
main.py, gui_app.py            CLI and desktop interfaces
multi_timeline.py              Parallel seeded runs (quantum_ghost_engine.py is a compatibility alias)
data_wrapper.py                Compatibility alias for engine_v3
spatiotemporal.py              Older 3D module, kept for compatibility
visualization_3d.py            Optional matplotlib voxel plot (for the older 3D module)
benchmark.py                   Synthetic benchmark vs simple baselines
real_benchmark.py              Real images vs scipy / scikit-image (needs requirements-dev.txt)
ablation_study.py              What each engine component contributes
benchmark_real_climate.py      Climate / vegetation time series (NetCDF, GeoTIFF, NumPy) under cloud masks
calibrate_confidence.py        Does the confidence score track real error?
research/                      evaluate_blends.py, make_repaired.py (need the temperature/humidity datasets)
tests/                         Unit and regression tests
.streamlit/config.toml         Upload limit and theme
```

## Tests

```bash
python -m unittest discover -s tests -t .        # 73 tests, about 1 s (needs numpy, pandas, scipy, scikit-image)
python tests/test_all_scenarios.py                # geometric scenario walkthrough (prints, no asserts)
```

## Benchmarks (reproduce, do not trust)

```bash
pip install -r requirements-dev.txt
python benchmark.py
python real_benchmark.py --size 64 --seeds 5
python ablation_study.py --markdown ablation.md
python calibrate_confidence.py --seeds 30
python benchmark_real_climate.py --data your_file.nc --var t2m --require-real
```

Without `--data`, `benchmark_real_climate.py` runs on a labelled synthetic field; those numbers are not
real-data results. Hidden values are never given to any method; they are used only for scoring.

## What it is not

It does not recover true hidden values. It produces auditable estimates, and the confidence score is a
heuristic: use the P95 error from the calibration report, not the score itself, as an error margin.
