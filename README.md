# Underwater Sound Propagation Simulator

**A cellular automaton that lets you *see* sound travel through the ocean.**

You drop a pressure source into a 2D slice of water with a realistic
depth-varying sound speed profile, hit play, and watch acoustic waves
bend, bounce, and interfere in real time — on your GPU if you have one.

Click anywhere on the grid and a probe pops up showing you exactly what
a hydrophone at that point would "hear," overlaid against the original
source signal so you can see how the ocean mangled it.

---

## Why is this fun?

Sound in the ocean doesn't travel in straight lines. The speed of sound
changes with depth (temperature, salinity, pressure), and that bends
the wavefronts like a lens. The surface reflects with a phase flip
(pressure release), the seabed reflects hard, and side boundaries can
be open or closed. All of this creates interference patterns, shadow
zones, and convergence zones — the same phenomena real sonar engineers
wrestle with — and you get to watch them form from scratch.

This simulator uses the **wave equation** (second-order in time and
space) solved on a uniform grid with explicit finite differences. It's
not a ray tracer or a parabolic approximation — it's a brute-force
full-field solver that naturally produces diffraction, multipath, and
interference without any special-case code.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Launch the real-time GUI (auto-selects GPU if available)
python scripts/gui.py

# 3. Or run a batch experiment that saves a GIF and summary plot
python scripts/run_experiment.py --config configs/default.yaml
```

That's it. The default config drops a 200 Hz source in a 160 m × 90 m
ocean slice with a realistic SSP and lets you watch.

---

## Project Layout

```
.
├── src/
│   └── pressure_transfer_ca.py   # The engine — physics, solvers, GPU kernels, plotting
│
├── scripts/
│   ├── gui.py                    # Real-time Pygame viewer with click-to-probe
│   ├── run_experiment.py         # Batch runner → timestamped experiment directories
│   ├── benchmark_backends.py     # Time CPU vs GPU backends
│   └── sweep.py                  # Parameter sweep helper
│
├── configs/                      # YAML experiment configs (copy, tweak, run)
│   ├── default.yaml              # Fully commented template — start here
│   ├── no_side_reflection.yaml   # Open left/right edges, 400 m wide
│   └── 24khz_all_reflect.yaml   # High-frequency small tank (3 m × 1.5 m)
│
├── experiments/                  # Output (git-ignored). One folder per run:
│   └── <timestamp>_<params>/
│       ├── config.yaml           # Exact config used (cloned for reproducibility)
│       ├── summary.png           # 4-panel snapshot (static P, SSP, final field, source)
│       └── propagation.gif       # Animated wave propagation
│
├── requirements.txt
├── .gitignore
└── README.md                     # You are here
```

---

## The Physics (explained simply)

### The wave equation on a grid

Imagine the ocean as a chessboard. Each cell holds a pressure value.
Every timestep, each cell looks at its four neighbours and updates
itself using the **discrete wave equation**:

```
p_next[y,x] = 2·p[y,x] − p_prev[y,x] + c(y,x)²·dt²·∇²p
```

where `∇²p` is the discrete Laplacian (sum of neighbour differences)
and `c(y,x)` is the local speed of sound. That's it — three lines of
maths produce everything: propagation, refraction, diffraction,
interference.

### Sound speed profile (SSP)

You give the simulator a list of `(depth, speed)` pairs and it
linearly interpolates them onto the grid. A typical ocean profile
looks like a bathtub curve — fast near the surface (warm), slow in
the thermocline, fast again at depth (high pressure). This bending
of the speed profile is what curves the wavefronts.

### Boundary conditions

Each wall (top/bottom/left/right) has a **reflection coefficient**
between −1 and +1:

| Value | Meaning |
|-------|---------|
| `+1.0` | Perfect rigid reflection (hard wall) |
| `−1.0` | Perfect phase-inverting reflection (pressure-release surface) |
| `0.0` | Fully open — energy leaves the domain |
| `−0.98` | Typical ocean surface (almost phase-inverting, slight loss) |
| `+0.99` | Typical seabed (almost rigid, slight loss) |

### Anti-aliasing (automatic)

A wave with frequency `f` and speed `c` has wavelength `λ = c / f`.
To avoid aliasing (garbage results), the grid cells must be smaller
than `λ / 5` at minimum, and the timestep must satisfy the CFL
condition. **You don't have to think about this.** Set `dx`, `dy`,
and `dt` to `0` in your config and the engine auto-computes them
from your frequency:

```yaml
grid:
  width_m: 400       # physical size — the only thing you specify
  height_m: 90
  dx: 0              # auto: λ / 10
  dy: 0              # auto: same as dx
  cells_per_wavelength: 10   # crank this up for more accuracy, down for speed

time:
  dt: 0              # auto: CFL-safe

source:
  frequency: 200     # Hz — this drives the auto-resolution
```

The `cells_per_wavelength` knob (default 10) controls the trade-off
between accuracy and grid size. At 10 cells/λ you get ~32
samples/cycle and clean waveforms. At 5 you save 4× the memory but
the waves get a bit lumpy.

---

## Interactive GUI

```bash
python scripts/gui.py --config configs/24khz_all_reflect.yaml
```

This opens a Pygame window showing the pressure field as a live
heatmap (blue = negative, white = zero, red = positive). The colour
scale auto-adapts to the current peak pressure.

### Controls

| Key / Action | What it does |
|--------------|--------------|
| **Space** | Play / pause |
| **→** | Step forward one frame (when paused) |
| **←** | Step backward (playback mode only) |
| **R** | Reset simulation to t = 0 |
| **+ / −** | Double / halve steps-per-frame (simulation speed) |
| **Left-click** on grid | Place a probe — its signal appears in the side panel |
| **Right-click** on grid marker | Remove that probe |
| **Click legend** entry | Remove that probe's plot |
| **Esc / Q** | Quit |

### Probe panel

The right side of the window shows a combined plot of all active
probes in different colours, with the analytical source sinusoid
drawn on a secondary axis for comparison. This tells you how the
ocean has transformed the signal at each receiver point.

### Modes

- **`--mode live`** (default): computes on the fly. Best for
  exploration. Press `+` to crank up speed — the engine batches
  thousands of GPU steps per frame.
- **`--mode playback`**: pre-computes the full simulation, then lets
  you scrub back and forth like a video.

### HUD

The bar at the bottom shows: current step, simulation time, FPS,
steps/sec throughput, backend name, steps-per-frame, colour scale,
and a controls reminder.

---

## Batch Experiments

```bash
python scripts/run_experiment.py --config configs/no_side_reflection.yaml
```

This runs the simulation to completion and saves everything into
`experiments/<timestamp>_<params>/`:

- **`config.yaml`** — exact copy of the config used (for
  reproducibility — you can re-run any old experiment from its own
  folder)
- **`summary.png`** — four-panel plot: static pressure, sound speed
  profile, final pressure field, source waveform
- **`propagation.gif`** — animated wave propagation

### CLI overrides

Any config parameter can be overridden from the command line without
editing the file:

```bash
# Quick test: 100 steps, no GIF
python scripts/run_experiment.py --steps 100 --no-gif

# Change frequency on the fly
python scripts/run_experiment.py --config configs/default.yaml --freq-hz 1000

# Override one boundary
python scripts/run_experiment.py --config configs/no_side_reflection.yaml --reflect-left 0.5
```

Priority: **CLI flag > config file > built-in default**.

### Key CLI options

| Flag | What | Default |
|------|------|---------|
| `--config` | YAML config file path | `configs/default.yaml` |
| `--nx`, `--ny` | Grid cell counts | auto from config |
| `--dx`, `--dy` | Cell size (m) | auto from frequency |
| `--dt` | Timestep (s), 0 = auto CFL | auto |
| `--steps` | Number of steps | from config |
| `--duration` | Total sim time (s) | from config |
| `--freq-hz` | Source frequency (Hz) | from config |
| `--backend` | `auto`, `cpu`, `gpu` | `auto` |
| `--no-gif` | Skip GIF generation | `false` |
| `--tag` | Label appended to output dir name | `""` |

---

## Config Files

All simulation parameters live in YAML files under `configs/`. The
`default.yaml` is exhaustively commented — read it first.

### Creating a new experiment

```bash
cp configs/default.yaml configs/my_experiment.yaml
# edit my_experiment.yaml
python scripts/run_experiment.py --config configs/my_experiment.yaml
```

### Included configs

| File | What it sets up |
|------|-----------------|
| `default.yaml` | 200 Hz source, 160 m × 90 m ocean, realistic SSP, all reflections on |
| `no_side_reflection.yaml` | 200 Hz, 400 m × 90 m, left/right edges fully open |
| `24khz_all_reflect.yaml` | 24 kHz, 3 m × 1.5 m lab tank, all walls reflecting |

### Config structure at a glance

```yaml
grid:
  width_m: 160          # Physical domain size (m)
  height_m: 90
  dx: 0                 # 0 = auto from frequency (recommended)
  dy: 0
  cells_per_wavelength: 10

time:
  dt: 0                 # 0 = auto CFL-safe (recommended)
  duration: 1.0         # seconds (or use steps: 1800)

source:
  frequency: 200        # Hz — THIS drives auto-resolution
  amplitude: 1400.0     # Pa
  ix: null              # null = center of grid
  iy: null

boundary:
  top:    -0.98         # Surface (phase-inverting)
  bottom:  0.99         # Seabed (rigid)
  left:    0.99         # Side wall
  right:   0.99

ssp:
  depths: [0, 15, 35, 60, 90]          # metres
  speeds: [1535, 1518, 1492, 1503, 1520]  # m/s

backend: auto           # auto, cpu, gpu
```

---

## Performance & GPU Acceleration

The simulator has three compute backends, selected automatically:

| Backend | What | When |
|---------|------|------|
| **GPU (CuPy + fused CUDA)** | Custom CUDA kernels for interior update + boundary in 2 launches per step | CuPy installed + NVIDIA GPU detected |
| **CPU (Numba)** | JIT-compiled parallel loops across all CPU cores | CuPy unavailable, Numba installed |
| **CPU (NumPy)** | Pure vectorized NumPy | Fallback |

### GPU pipeline

The `step_n()` method batches N timesteps on the GPU with:

- **Zero intermediate GPU→CPU transfers** — the field stays on the
  GPU for all N steps, transferred to CPU only once at the end for
  display
- **Fused CUDA boundary kernel** — boundaries + source injection in
  a single kernel launch (not 5 separate CuPy operations)
- **Probe subsampling** — probe values collected every Kth step on
  GPU (configurable stride), not every step
- **Vectorized source** — all N source signal values pre-computed in
  one NumPy `sin()` call

This gives **~20× throughput** vs naive per-step GPU calls.

### Benchmarking

```bash
python scripts/benchmark_backends.py --model wave --steps 1200 --nx 220 --ny 140
```

---

## How the code is structured

### `src/pressure_transfer_ca.py` — the engine

Everything lives in one file (it's a simulation engine, not a web
app). Key pieces:

| Thing | What |
|-------|------|
| `PressureCASimConfig` | Dataclass holding every simulation parameter |
| `auto_resolve_grid()` | Computes dx/dy/dt/nx/ny from frequency + domain size |
| `build_sound_speed_grid()` | Interpolates SSP onto the grid |
| `WaveStepper` | Stateful stepper for interactive use (GUI). Picks backend, exposes `step()` and `step_n()` |
| `run_pressure_transfer_ca()` | One-shot batch runner (experiments). Returns dict with fields, frames, traces |
| `plot_static_and_final()` | Four-panel summary figure |
| `save_gif_parallel()` | Multi-process GIF renderer |
| CUDA kernels | `wave_step_f32/f64` (interior), `wave_boundary_f32/f64` (boundaries + source) |
| Numba kernel | `_build_numba_wave_kernel()` — parallel CPU equivalent |

### `scripts/gui.py` — the viewer

Built on Pygame. Architecture:

- **Main loop**: event handling → `stepper.step_n()` → colour-map
  the field → blit to screen → render HUD
- **Probes**: `Probe` objects record `(time, pressure)`. A background
  thread renders the Matplotlib plot to an image buffer so the UI
  never freezes.
- **Layout**: dynamically recalculated on window resize. Grid on the
  left, probe panel on the right, HUD bar at the bottom.

### `scripts/run_experiment.py` — the batch runner

Loads config → calls `run_pressure_transfer_ca()` → saves summary
plot + GIF → clones config into output dir.

---

## Requirements

- Python 3.10+
- NumPy, SciPy, Matplotlib, Pillow, PyYAML, Pygame
- **Optional**: CuPy (GPU), Numba (fast CPU)

```bash
pip install -r requirements.txt

# For GPU acceleration (needs CUDA toolkit):
pip install cupy-cuda12x   # or cupy-cuda11x for older GPUs
```

---

## FAQ

**Q: The simulation is slow / the grid is huge.**
A: Lower `cells_per_wavelength` to 5–7, or reduce the physical domain
size, or lower the frequency. High frequencies need tiny cells.
A 24 kHz source in a 400 m ocean needs ~64,000 × 14,000 cells — use
the small tank config instead.

**Q: The wave looks blocky / aliased.**
A: Increase `cells_per_wavelength` or make sure `dx`/`dy`/`dt` are
set to `0` (auto). If you hardcoded cell sizes that are too large
for your frequency, the engine will warn you but won't stop you.

**Q: Can I use this for real research?**
A: It's a teaching / exploration tool. The physics are correct (2nd
order wave equation, depth-varying SSP, boundary reflections) but
it's 2D, uses uniform grids, and doesn't model 3D spreading loss,
frequency-dependent absorption, or elastic bottom layers. For
publication-grade work, look at KRAKEN, Bellhop, or RAM.

**Q: Why not just use a ray tracer?**
A: Ray tracers are fast but miss diffraction and low-frequency
effects. This full-field solver naturally captures everything — at
the cost of computing every grid cell every timestep. That's what
the GPU is for.

---

*Part of the SMaRC PhD project — underwater acoustics meets cellular automata.*
