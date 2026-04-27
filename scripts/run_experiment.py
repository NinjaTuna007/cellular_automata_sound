#!/usr/bin/env python3
"""Batch experiment runner for the underwater sound simulator.

Runs a full simulation from a YAML config file, then saves all output
into a timestamped directory under ``experiments/``.

Workflow
-------
::

    cp configs/default.yaml configs/my_experiment.yaml
    # edit my_experiment.yaml
    python scripts/run_experiment.py --config configs/my_experiment.yaml

What you get in the output directory:

- ``config.yaml`` — exact clone of the config used (so you can always
  reproduce the run from its own folder, even years later)
- ``summary.png`` — four-panel plot: hydrostatic pressure, sound speed
  profile, final acoustic field, source waveform
- ``propagation.gif`` — animated wave propagation (skip with ``--no-gif``)

CLI flags override config values for quick one-off tweaks without
editing the file (priority: CLI > config > built-in defaults).

Examples
--------
::

    python scripts/run_experiment.py                          # default config
    python scripts/run_experiment.py --steps 100 --no-gif     # quick smoke test
    python scripts/run_experiment.py --freq-hz 1000           # change frequency
    python scripts/run_experiment.py --reflect-left 0.0       # open one wall
"""
from __future__ import annotations

import argparse
import shutil
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import time as _time

from tqdm import tqdm

from pressure_transfer_ca import (
    PressureCASimConfig,
    auto_resolve_grid,
    plot_static_and_final,
    run_pressure_transfer_ca,
    save_gif_parallel,
)

from wave_3d import (
    WaveConfig3D,
    auto_resolve_grid_3d,
    run_wave_3d,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run a pressure CA experiment from a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python scripts/run_experiment.py --config configs/no_side_reflection.yaml
              python scripts/run_experiment.py --config configs/default.yaml --no-gif
              python scripts/run_experiment.py  # uses configs/default.yaml
        """),
    )
    p.add_argument("--config", type=str, default=None,
                    help="Path to YAML config file (default: configs/default.yaml)")

    over = p.add_argument_group("CLI overrides (applied on top of config)")
    over.add_argument("--nx", type=int, default=None)
    over.add_argument("--ny", type=int, default=None)
    over.add_argument("--dx", type=float, default=None)
    over.add_argument("--dy", type=float, default=None)
    over.add_argument("--dt", type=float, default=None)
    over.add_argument("--steps", type=int, default=None)
    over.add_argument("--duration", type=float, default=None)
    over.add_argument("--model", type=str, default=None, choices=["wave", "transfer"])
    over.add_argument("--freq-hz", type=float, default=None)
    over.add_argument("--source-amplitude", type=float, default=None)
    over.add_argument("--source-ix", type=int, default=None)
    over.add_argument("--source-iy", type=int, default=None)
    over.add_argument("--absorption", type=float, default=None)
    over.add_argument("--reflect-top", type=float, default=None)
    over.add_argument("--reflect-bottom", type=float, default=None)
    over.add_argument("--reflect-left", type=float, default=None)
    over.add_argument("--reflect-right", type=float, default=None)
    over.add_argument("--backend", type=str, default=None, choices=["auto", "cpu", "gpu"])
    over.add_argument("--frame-stride", type=int, default=None)
    over.add_argument("--gif-fps", type=int, default=None)
    over.add_argument("--no-gif", action="store_true", default=None)
    over.add_argument("--no-frames", action="store_true", default=None,
                       help="Skip saving frame data for GUI replay")
    over.add_argument("--no-grid", action="store_true", default=None,
                       help="Hide cell boundary grid lines")
    over.add_argument("--tag", type=str, default=None)
    over.add_argument("--3d", dest="is_3d", action="store_true", default=False,
                       help="Run 3D simulation (requires 3D config)")
    over.add_argument("--depth-m", type=float, default=None,
                       help="(3D only) domain depth in metres")

    return p.parse_args()


def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def get(cfg: dict, *keys, default=None):
    """Nested dict lookup: get(cfg, 'boundary', 'top', default=0.0)"""
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


def parse_range_list(raw) -> tuple:
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    ranges = []
    for token in raw:
        token = str(token).strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid range '{token}'. Expected start:end")
        s, e = token.split(":", 1)
        ranges.append((int(s), int(e)))
    return tuple(ranges)


def config_to_sim(cfg: dict, args) -> PressureCASimConfig:
    """Build PressureCASimConfig from YAML dict + CLI overrides."""

    ssp_depths = tuple(get(cfg, "ssp", "depths", default=[0, 15, 35, 60, 90]))
    ssp_speeds = tuple(get(cfg, "ssp", "speeds", default=[1535, 1518, 1492, 1503, 1520]))
    freq = args.freq_hz if args.freq_hz is not None else get(cfg, "source", "frequency", default=24000)

    stencil_order = get(cfg, "grid", "stencil_order", default=2)
    grid = auto_resolve_grid(
        frequency_hz=freq,
        c_min=min(ssp_speeds),
        c_max=max(ssp_speeds),
        width_m=get(cfg, "grid", "width_m", default=None),
        height_m=get(cfg, "grid", "height_m", default=None),
        dx=args.dx or get(cfg, "grid", "dx", default=0),
        dy=args.dy or get(cfg, "grid", "dy", default=0),
        dt=args.dt if args.dt is not None else get(cfg, "time", "dt", default=0),
        nx=args.nx or get(cfg, "grid", "nx", default=0),
        ny=args.ny or get(cfg, "grid", "ny", default=0),
        cells_per_wavelength=get(cfg, "grid", "cells_per_wavelength", default=0),
        stencil_order=stencil_order,
    )
    nx, ny = grid["nx"], grid["ny"]
    dx, dy, dt = grid["dx"], grid["dy"], grid["dt"]

    steps_raw = args.steps if args.steps is not None else get(cfg, "time", "steps", default=1800)
    steps_explicit_cli = args.steps is not None
    if args.duration is not None:
        duration = args.duration
    elif steps_explicit_cli:
        duration = 0
    else:
        duration = get(cfg, "time", "duration", default=0)

    if duration and duration > 0:
        steps = max(1, int(round(duration / dt)))
    else:
        steps = steps_raw

    model = args.model or get(cfg, "model", "type", default="wave")
    absorption = args.absorption if args.absorption is not None else get(cfg, "model", "absorption", default=0.05)

    source_ix_raw = args.source_ix if args.source_ix is not None else get(cfg, "source", "ix", default=None)
    source_iy_raw = args.source_iy if args.source_iy is not None else get(cfg, "source", "iy", default=None)
    src_x = get(cfg, "source", "x", default=None)
    src_y = get(cfg, "source", "y", default=None)
    if src_x is not None and source_ix_raw is None:
        source_ix_raw = int(round(float(src_x) / dx))
    if src_y is not None and source_iy_raw is None:
        source_iy_raw = int(round(float(src_y) / dy))
    source_ix = source_ix_raw if source_ix_raw is not None else nx // 2
    source_iy = source_iy_raw if source_iy_raw is not None else ny // 2
    source_ix = max(0, min(source_ix, nx - 1))
    source_iy = max(0, min(source_iy, ny - 1))
    amplitude = args.source_amplitude if args.source_amplitude is not None else get(cfg, "source", "amplitude", default=1400.0)

    r_top = args.reflect_top if args.reflect_top is not None else get(cfg, "boundary", "top", default=-0.98)
    r_bot = args.reflect_bottom if args.reflect_bottom is not None else get(cfg, "boundary", "bottom", default=0.99)
    r_lft = args.reflect_left if args.reflect_left is not None else get(cfg, "boundary", "left", default=0.99)
    r_rgt = args.reflect_right if args.reflect_right is not None else get(cfg, "boundary", "right", default=0.99)

    backend = args.backend or get(cfg, "backend", default="auto")
    use_float32 = get(cfg, "use_float32", default=True)
    frame_stride = args.frame_stride if args.frame_stride is not None else get(cfg, "output", "frame_stride", default=2)

    return PressureCASimConfig(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, steps=steps,
        propagation_model=model,
        stencil_order=stencil_order,
        transfer_fraction=get(cfg, "transfer", "fraction", default=0.42),
        damping=get(cfg, "transfer", "damping", default=0.002),
        diagonal_interface_scale=get(cfg, "transfer", "diagonal_interface_scale", default=0.35),
        overpressure_only=get(cfg, "transfer", "overpressure_only", default=False),
        refraction_strength=get(cfg, "transfer", "refraction_strength", default=0.45),
        use_impedance_interface=get(cfg, "transfer", "use_impedance_interface", default=True),
        wave_absorption_per_s=absorption,
        boundary_reflect_top=r_top,
        boundary_reflect_bottom=r_bot,
        boundary_reflect_left=r_lft,
        boundary_reflect_right=r_rgt,
        top_open_x_ranges=parse_range_list(get(cfg, "boundary", "top_open_x", default=[])),
        bottom_open_x_ranges=parse_range_list(get(cfg, "boundary", "bottom_open_x", default=[])),
        left_open_y_ranges=parse_range_list(get(cfg, "boundary", "left_open_y", default=[])),
        right_open_y_ranges=parse_range_list(get(cfg, "boundary", "right_open_y", default=[])),
        backend=backend,
        use_float32=use_float32,
        source_ix=source_ix,
        source_iy=source_iy,
        source_amplitude_pa=amplitude,
        source_frequency_hz=freq,
        ssp_depths_m=ssp_depths,
        ssp_speeds_mps=ssp_speeds,
        frame_stride=frame_stride,
    )


def config_to_sim_3d(cfg: dict, args) -> WaveConfig3D:
    """Build WaveConfig3D from YAML dict + CLI overrides."""
    ssp_depths = tuple(get(cfg, "ssp", "depths", default=[0, 5]))
    ssp_speeds = tuple(get(cfg, "ssp", "speeds", default=[1500, 1500]))
    freq = args.freq_hz if args.freq_hz is not None else get(cfg, "source", "frequency", default=1000)
    stencil_order = get(cfg, "grid", "stencil_order", default=2)

    grid = auto_resolve_grid_3d(
        frequency_hz=freq,
        c_min=min(ssp_speeds), c_max=max(ssp_speeds),
        width_m=get(cfg, "grid", "width_m"),
        height_m=get(cfg, "grid", "height_m"),
        depth_m=args.depth_m or get(cfg, "grid", "depth_m"),
        dx=args.dx or get(cfg, "grid", "dx", default=0),
        dy=args.dy or get(cfg, "grid", "dy", default=0),
        dz=get(cfg, "grid", "dz", default=0),
        dt=args.dt if args.dt is not None else get(cfg, "time", "dt", default=0),
        nx=args.nx or get(cfg, "grid", "nx", default=0),
        ny=args.ny or get(cfg, "grid", "ny", default=0),
        nz=get(cfg, "grid", "nz", default=0),
        cells_per_wavelength=get(cfg, "grid", "cells_per_wavelength", default=0),
        stencil_order=stencil_order,
    )
    nx, ny, nz = grid["nx"], grid["ny"], grid["nz"]
    dx, dy, dz, dt_val = grid["dx"], grid["dy"], grid["dz"], grid["dt"]

    steps_raw = args.steps if args.steps is not None else get(cfg, "time", "steps", default=100000)
    duration = get(cfg, "time", "duration", default=0)
    if args.duration is not None:
        duration = args.duration
    if duration and duration > 0:
        steps = max(1, int(round(duration / dt_val)))
    else:
        steps = steps_raw

    absorption = args.absorption if args.absorption is not None else get(cfg, "model", "absorption", default=0.05)
    src_ix = args.source_ix if args.source_ix is not None else get(cfg, "source", "ix")
    src_iy = args.source_iy if args.source_iy is not None else get(cfg, "source", "iy")
    src_iz = get(cfg, "source", "iz")
    # Physical coords (metres) take priority over cell indices
    src_x = get(cfg, "source", "x")
    src_y = get(cfg, "source", "y")
    src_z = get(cfg, "source", "z")
    if src_x is not None:
        src_ix = int(round(float(src_x) / dx))
    if src_y is not None:
        src_iy = int(round(float(src_y) / dy))
    if src_z is not None:
        src_iz = int(round(float(src_z) / dz))
    if src_ix is None:
        src_ix = nx // 2
    if src_iy is None:
        src_iy = ny // 2
    if src_iz is None:
        src_iz = nz // 2
    src_ix = max(0, min(src_ix, nx - 1))
    src_iy = max(0, min(src_iy, ny - 1))
    src_iz = max(0, min(src_iz, nz - 1))

    amplitude = args.source_amplitude if args.source_amplitude is not None else get(cfg, "source", "amplitude", default=500.0)
    backend = args.backend or get(cfg, "backend", default="auto")
    frame_stride = args.frame_stride if args.frame_stride is not None else get(cfg, "output", "frame_stride", default=5)

    return WaveConfig3D(
        nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz, dt=dt_val, steps=steps,
        ssp_depths_m=ssp_depths, ssp_speeds_mps=ssp_speeds,
        wave_absorption_per_s=absorption,
        boundary_reflect_top=args.reflect_top if args.reflect_top is not None else get(cfg, "boundary", "top", default=-0.98),
        boundary_reflect_bottom=args.reflect_bottom if args.reflect_bottom is not None else get(cfg, "boundary", "bottom", default=0.99),
        boundary_reflect_left=args.reflect_left if args.reflect_left is not None else get(cfg, "boundary", "left", default=0.99),
        boundary_reflect_right=args.reflect_right if args.reflect_right is not None else get(cfg, "boundary", "right", default=0.99),
        boundary_reflect_front=get(cfg, "boundary", "front", default=0.0),
        boundary_reflect_back=get(cfg, "boundary", "back", default=0.0),
        stencil_order=stencil_order,
        backend=backend,
        use_float32=get(cfg, "use_float32", default=True),
        source_ix=src_ix, source_iy=src_iy, source_iz=src_iz,
        source_amplitude_pa=amplitude,
        source_frequency_hz=freq,
        frame_stride=frame_stride,
    )


def build_experiment_dirname(sim_cfg, tag: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if tag:
        return f"{ts}_{tag}"
    return ts


MAX_FRAME_DIM = 2048  # downsample frames if either dimension exceeds this


def _save_frames(frames_raw: list, sim_cfg: PressureCASimConfig, exp_dir: Path):
    """Compress and save simulation frames for later GUI replay.

    Large grids are spatially downsampled so the .npz stays manageable.
    """
    ny, nx = frames_raw[0].shape
    ds = 1  # downsample factor
    while ny // ds > MAX_FRAME_DIM or nx // ds > MAX_FRAME_DIM:
        ds += 1

    if ds > 1:
        print(f"Downsampling frames {ds}× for storage ({nx}×{ny} → {nx // ds}×{ny // ds})")
        down = []
        for f in frames_raw:
            trimmed = f[:ny - ny % ds, :nx - nx % ds]
            down.append(trimmed.reshape(trimmed.shape[0] // ds, ds,
                                        trimmed.shape[1] // ds, ds).mean(axis=(1, 3)))
        frames_arr = np.array(down, dtype=np.float32)
    else:
        frames_arr = np.array(frames_raw, dtype=np.float32)

    frame_times = np.arange(len(frames_arr)) * sim_cfg.dt * max(1, sim_cfg.frame_stride)
    npz_path = exp_dir / "frames.npz"
    np.savez_compressed(
        npz_path,
        frames=frames_arr,
        frame_times=frame_times,
        dt=sim_cfg.dt,
        frame_stride=sim_cfg.frame_stride,
        nx=nx, ny=ny,
        dx=sim_cfg.dx, dy=sim_cfg.dy,
        ds_factor=ds,
        source_frequency_hz=sim_cfg.source_frequency_hz,
        stencil_order=sim_cfg.stencil_order,
        source_ix=sim_cfg.source_ix,
        source_iy=sim_cfg.source_iy,
    )
    mb = npz_path.stat().st_size / (1024 * 1024)
    print(f"Saved          : {npz_path.name} ({len(frames_arr)} frames, {mb:.1f} MB)")


def _run_3d_experiment(args, cfg, config_path):
    """Run a 3D simulation experiment."""
    sim_cfg = config_to_sim_3d(cfg, args)
    tag = args.tag if args.tag is not None else get(cfg, "output", "tag", default="")

    dirname = build_experiment_dirname(sim_cfg, tag)
    exp_dir = PROJECT_ROOT / "experiments" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, exp_dir / "config.yaml")

    total_cells = sim_cfg.nx * sim_cfg.ny * sim_cfg.nz
    print(f"Experiment dir : {exp_dir}")
    print(f"3D Grid        : {sim_cfg.nx}x{sim_cfg.ny}x{sim_cfg.nz} = "
          f"{total_cells / 1e6:.2f}M cells")
    print(f"Cell size      : dx={sim_cfg.dx:.4f} dy={sim_cfg.dy:.4f} dz={sim_cfg.dz:.4f}")
    print(f"Time           : dt={sim_cfg.dt:.6e} s, {sim_cfg.steps} steps, "
          f"total={sim_cfg.steps * sim_cfg.dt:.4f} s")
    print(f"Source         : ({sim_cfg.source_ix}, {sim_cfg.source_iy}, {sim_cfg.source_iz}), "
          f"freq={sim_cfg.source_frequency_hz} Hz")
    print(f"Boundaries     : top={sim_cfg.boundary_reflect_top}, "
          f"bot={sim_cfg.boundary_reflect_bottom}, "
          f"L={sim_cfg.boundary_reflect_left}, R={sim_cfg.boundary_reflect_right}, "
          f"F={sim_cfg.boundary_reflect_front}, B={sim_cfg.boundary_reflect_back}")

    n_frames_est = sim_cfg.steps // max(1, sim_cfg.frame_stride) + 1
    print(f"Frames         : ~{n_frames_est} (stride {sim_cfg.frame_stride})")

    print("Running 3D simulation...")
    pbar = tqdm(total=sim_cfg.steps, unit="step", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def _tqdm_progress(step: int, total: int):
        pbar.update(step - pbar.n)

    t0 = _time.monotonic()
    result = run_wave_3d(sim_cfg, progress_fn=_tqdm_progress,
                         frame_dir=str(exp_dir))
    elapsed = _time.monotonic() - t0
    pbar.close()
    backend_used = result.get("backend_used", "unknown")
    print(f"Backend        : {backend_used}")
    print(f"Wall time      : {elapsed:.1f} s "
          f"({sim_cfg.steps / max(0.01, elapsed):.0f} steps/s)")

    no_frames = args.no_frames if args.no_frames else get(cfg, "output", "no_frames", default=False)
    sink = result.get("_frame_sink")

    if no_frames:
        if sink is not None:
            sink.cleanup()
    elif sink is not None:
        if sink._mode == "ram":
            frames = result.get("frames", [])
            if frames and len(frames) > 0:
                frame_times = np.arange(len(frames)) * sim_cfg.dt * max(1, sim_cfg.frame_stride)
                npz_path = exp_dir / "frames.npz"
                frame_arr = np.array([np.asarray(f) for f in frames], dtype=np.float32)
                np.savez_compressed(
                    npz_path, frames=frame_arr, frame_times=frame_times,
                    dt=sim_cfg.dt, frame_stride=sim_cfg.frame_stride,
                    nx=sim_cfg.nx, ny=sim_cfg.ny, nz=sim_cfg.nz,
                    dx=sim_cfg.dx, dy=sim_cfg.dy, dz=sim_cfg.dz,
                    source_ix=sim_cfg.source_ix, source_iy=sim_cfg.source_iy,
                    source_iz=sim_cfg.source_iz,
                    source_frequency_hz=sim_cfg.source_frequency_hz,
                    stencil_order=sim_cfg.stencil_order,
                )
                mb = npz_path.stat().st_size / (1024 * 1024)
                print(f"Saved          : {npz_path.name} ({len(frames)} frames, {mb:.1f} MB)")
            sink.cleanup()
        else:
            # Disk-backed: keep cache files for gui_3d replay
            if sink._fh and not sink._fh.closed:
                sink._fh.close()
            cache_mb = (sink.count * sink._frame_nbytes) / (1024 * 1024)
            print(f"Replay data    : {sink.count} frames in {sink.disk_path} ({cache_mb:.0f} MB)")

    print(f"\nDone. Results in:\n  {exp_dir}")


def main():
    args = parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config file not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    cfg = load_config(config_path)

    # ── 3D mode ──
    if args.is_3d:
        _run_3d_experiment(args, cfg, config_path)
        return

    # ── 2D mode (existing) ──
    sim_cfg = config_to_sim(cfg, args)

    no_gif = args.no_gif if args.no_gif else get(cfg, "output", "no_gif", default=False)
    no_grid = args.no_grid if args.no_grid else get(cfg, "output", "no_grid", default=False)
    gif_fps = args.gif_fps if args.gif_fps is not None else get(cfg, "output", "gif_fps", default=25)
    tag = args.tag if args.tag is not None else get(cfg, "output", "tag", default="")

    dirname = build_experiment_dirname(sim_cfg, tag)
    exp_dir = PROJECT_ROOT / "experiments" / dirname
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Clone config into experiment directory for reproducibility.
    shutil.copy2(config_path, exp_dir / "config.yaml")

    print(f"Experiment dir : {exp_dir}")
    print(f"Grid           : {sim_cfg.nx}x{sim_cfg.ny}, dx={sim_cfg.dx} dy={sim_cfg.dy}")
    print(f"Time           : dt={sim_cfg.dt:.6f} s, {sim_cfg.steps} steps, "
          f"total={sim_cfg.steps * sim_cfg.dt:.4f} s")
    print(f"Source         : ({sim_cfg.source_ix}, {sim_cfg.source_iy}), "
          f"freq={sim_cfg.source_frequency_hz} Hz, amp={sim_cfg.source_amplitude_pa} Pa")
    print(f"Boundaries     : top={sim_cfg.boundary_reflect_top}, "
          f"bottom={sim_cfg.boundary_reflect_bottom}, "
          f"left={sim_cfg.boundary_reflect_left}, "
          f"right={sim_cfg.boundary_reflect_right}")

    n_frames_est = sim_cfg.steps // max(1, sim_cfg.frame_stride) + 1
    print(f"Frames         : ~{n_frames_est} (stride {sim_cfg.frame_stride})")

    print("Running simulation...")
    pbar = tqdm(total=sim_cfg.steps, unit="step", dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

    def _tqdm_progress(step: int, total: int):
        pbar.update(step - pbar.n)

    t0 = _time.monotonic()
    result = run_pressure_transfer_ca(sim_cfg, progress_fn=_tqdm_progress,
                                       frame_dir=str(exp_dir))
    elapsed = _time.monotonic() - t0
    pbar.close()
    backend_used = result.get("backend_used", "unknown")
    print(f"Backend        : {backend_used}")
    print(f"Wall time      : {elapsed:.1f} s "
          f"({sim_cfg.steps / max(0.01, elapsed):.0f} steps/s)")

    show_grid = not no_grid
    no_frames = args.no_frames if args.no_frames else get(cfg, "output", "no_frames", default=False)

    fig = plot_static_and_final(result, show_grid=show_grid)
    png_path = exp_dir / "summary.png"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved          : {png_path.name}")

    if not no_gif:
        print("Generating GIF (parallel)...")
        gif_dpi = max(140, min(250, sim_cfg.nx // 2))
        gif_path = exp_dir / "propagation.gif"
        save_gif_parallel(result, str(gif_path), fps=gif_fps, dpi=gif_dpi,
                          show_time=True, show_grid=show_grid)
        print(f"Saved          : {gif_path.name}")

    # ── Save frame data for GUI replay ──
    if not no_frames:
        frames_raw = result.get("frames_extra", [])
        if frames_raw:
            _save_frames(frames_raw, sim_cfg, exp_dir)

    # Clean up temp memmap file if the frame sink used disk
    sink = result.get("_frame_sink")
    if sink is not None:
        sink.cleanup()

    print(f"\nDone. Results in:\n  {exp_dir}")


if __name__ == "__main__":
    main()
