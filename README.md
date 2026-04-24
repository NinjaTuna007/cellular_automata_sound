# Pressure Transfer Cellular Automaton

2D cellular automaton for underwater acoustic pressure propagation with depth-varying sound speed profiles.

## Project Structure

```
├── src/
│   └── pressure_transfer_ca.py    # Core engine: config, simulation, plotting
├── scripts/
│   ├── run_experiment.py           # Main experiment runner
│   └── benchmark_backends.py       # CPU vs GPU timing comparison
├── configs/
│   ├── default.yaml                # Fully documented template config
│   └── no_side_reflection.yaml     # Example: open left/right boundaries
├── experiments/                    # Each run creates a timestamped subdirectory
│   └── <timestamp>_<params>/
│       ├── config.yaml             # Cloned config (exact copy used for this run)
│       ├── summary.png             # 4-panel pressure snapshot
│       └── propagation.gif         # Animated wave propagation
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Running Experiments

### Config-driven workflow

1. Copy `configs/default.yaml` to a new file under `configs/`
2. Edit the parameters you care about
3. Run it:

```bash
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

The config file is **cloned verbatim** into the experiment output directory, so every result is self-contained and reproducible from its own folder.

### CLI overrides

CLI flags override config values for quick tweaks without editing the file:

```bash
python scripts/run_experiment.py --config configs/default.yaml --steps 500 --no-gif
```

Priority: CLI flag > config file > built-in default.

### Examples

```bash
# Run with default config
python scripts/run_experiment.py

# No side reflections, 400m wide, 1s duration
python scripts/run_experiment.py --config configs/no_side_reflection.yaml

# Quick smoke test (override steps, skip GIF)
python scripts/run_experiment.py --config configs/default.yaml --steps 100 --no-gif

# Override one boundary on top of a config
python scripts/run_experiment.py --config configs/no_side_reflection.yaml --reflect-left 0.5
```

### Key CLI options

| Option | Description | Default |
|--------|-------------|---------|
| `--config` | Path to YAML config file | `configs/default.yaml` |
| `--nx`, `--ny` | Grid cells | from config |
| `--dx`, `--dy` | Cell size (m) | from config |
| `--dt` | Time step (s), <=0 for auto CFL-safe | from config |
| `--steps` | Number of steps (overrides duration in config) | from config |
| `--duration` | Total sim time (s), overrides steps | from config |
| `--model` | `wave` or `transfer` | from config |
| `--freq-hz` | Source frequency (Hz) | from config |
| `--reflect-top/bottom/left/right` | Boundary reflection coefficients | from config |
| `--backend` | `auto`, `cpu`, `gpu` | from config |
| `--no-gif` | Skip GIF generation | from config |
| `--tag` | Extra label in experiment dir name | from config |

## Interactive GUI Viewer

Real-time Pygame viewer with click-to-probe pressure time-series.

```bash
# Live mode: compute and render on the fly
python scripts/gui.py --config configs/no_side_reflection.yaml

# Playback mode: pre-compute, then scrub/pause/rewind
python scripts/gui.py --config configs/no_side_reflection.yaml --mode playback

# Force a specific backend
python scripts/gui.py --config configs/default.yaml --backend gpu
```

**Controls:**

| Key | Action |
|-----|--------|
| Space | Play / pause |
| Right arrow | Step forward (when paused) |
| Left arrow | Step backward (playback mode) |
| R | Reset to step 0 |
| +/- | Speed up / slow down (steps per frame) |
| Click cell | Open probe window (pressure vs source time-series) |
| Esc / Q | Quit |

## Benchmarking

```bash
python scripts/benchmark_backends.py --model wave --steps 1200 --nx 220 --ny 140
```

## Physics Summary

- **Wave model** (recommended): second-order explicit update with depth-varying c(y), boundary reflections (surface phase inversion, seabed/side reflection), absorption.
- **Transfer model** (legacy): CA overpressure transfer with inverse-square weighting, impedance interfaces, refraction bias.
- SSP (sound speed profile) interpolated from user-defined depth/speed points.
- CFL stability enforced automatically for the wave model.
