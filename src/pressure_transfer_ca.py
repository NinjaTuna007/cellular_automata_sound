"""2D acoustic pressure propagation engine using cellular automata / finite differences.

This module is the core of the underwater sound simulator.  It solves the
second-order wave equation on a uniform 2D grid with a depth-varying sound
speed profile (SSP), configurable boundary reflections, and a sinusoidal
pressure source.

Three compute backends are supported (selected automatically):

1. **GPU** — fused CUDA kernels via CuPy (fastest, needs NVIDIA GPU)
2. **CPU-Numba** — JIT-compiled parallel loops (fast, needs Numba)
3. **CPU-NumPy** — pure vectorised NumPy (always available)

The two main entry points are:

- ``WaveStepper`` — stateful, one-step-at-a-time interface for the GUI.
  Supports batched ``step_n()`` to keep the GPU busy.
- ``run_pressure_transfer_ca()`` — one-shot batch run that returns the
  full frame history, source trace, and final field.

Both are configured via ``PressureCASimConfig``, a dataclass that holds
every knob.  Use ``auto_resolve_grid()`` to compute grid sizes and timestep
from the source frequency so you never have to worry about aliasing.

Plotting helpers (``plot_static_and_final``, ``save_gif_parallel``) are
included so experiments produce self-contained visual output.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:  # pragma: no cover - optional runtime backend
    cp = None  # type: ignore
    CUPY_AVAILABLE = False

try:
    import numba as nb

    _ncpu = os.cpu_count() or 4
    _nb_threads = max(1, _ncpu - 2)
    nb.config.NUMBA_NUM_THREADS = min(_nb_threads, nb.config.NUMBA_NUM_THREADS)
    NUMBA_AVAILABLE = True
except (ImportError, AttributeError):
    nb = None  # type: ignore
    NUMBA_AVAILABLE = False


@dataclass
class PressureCASimConfig:
    """Every parameter the simulator needs, in one place.

    Grid geometry, time stepping, physics (SSP, boundaries, absorption),
    source definition, and execution options.  Typically built by
    ``config_to_sim()`` from a YAML file, not constructed by hand.

    Set ``dx``, ``dy``, ``dt`` to 0 and call ``auto_resolve_grid()``
    to have them computed from the source frequency automatically.
    """

    # Grid resolution (number of cells)
    nx: int = 120
    ny: int = 80

    # Cell size (meters): rectangular cells are fully supported
    dx: float = 2.0
    dy: float = 1.0

    # Time stepping
    dt: float = 0.01
    steps: int = 500

    # Fluid properties for static pressure p_static = rho * g * h
    rho: float = 1000.0
    g: float = 9.81

    # Depth-dependent sound speed profile c(y), linearly interpolated by depth.
    ssp_depths_m: Tuple[float, ...] = (0.0, 20.0, 50.0, 80.0)
    ssp_speeds_mps: Tuple[float, ...] = (1525.0, 1515.0, 1498.0, 1510.0)

    # Propagation model: "wave" (recommended) or "transfer" (legacy toy rule)
    propagation_model: str = "wave"

    # Legacy CA transfer behavior (used when propagation_model == "transfer")
    transfer_fraction: float = 0.35
    damping: float = 0.01
    diagonal_interface_scale: float = 0.4
    overpressure_only: bool = False
    refraction_strength: float = 0.45
    use_impedance_interface: bool = True

    # Wave-model controls (used when propagation_model == "wave")
    wave_absorption_per_s: float = 0.15
    # Reflection coefficients at boundaries.
    # Surface defaults to pressure-release style (phase inversion).
    boundary_reflect_top: float = -0.98
    boundary_reflect_bottom: float = 0.98
    boundary_reflect_left: float = 0.98
    boundary_reflect_right: float = 0.98
    # Optional open (non-reflective) boundary segments in index ranges [start, end).
    # Example: top_open_x_ranges=((20, 40), (70, 90))
    top_open_x_ranges: Tuple[Tuple[int, int], ...] = ()
    bottom_open_x_ranges: Tuple[Tuple[int, int], ...] = ()
    left_open_y_ranges: Tuple[Tuple[int, int], ...] = ()
    right_open_y_ranges: Tuple[Tuple[int, int], ...] = ()

    # Execution backends: "auto", "cpu", "gpu"
    backend: str = "auto"
    # Numeric dtype for simulation arrays.
    use_float32: bool = True

    # Source forcing (extra pressure, on top of static baseline)
    source_ix: int = 8
    source_iy: int = 20
    source_amplitude_pa: float = 3500.0
    source_frequency_hz: float = 1.5
    source_phase_rad: float = 0.0

    # Frame sampling for visualization
    frame_stride: int = 2


def _neighbor_offsets() -> List[Tuple[int, int]]:
    return [
        (-1, -1), (0, -1), (1, -1),
        (-1, 0),           (1, 0),
        (-1, 1),  (0, 1),  (1, 1),
    ]


def _neighbor_distance(dx: float, dy: float, ox: int, oy: int) -> float:
    return float(np.sqrt((ox * dx) ** 2 + (oy * dy) ** 2))


def _effective_interface_length(dx: float, dy: float, ox: int, oy: int, diagonal_scale: float) -> float:
    if ox == 0 and oy != 0:
        return dx
    if oy == 0 and ox != 0:
        return dy
    return diagonal_scale * np.sqrt(dx * dy)


def _neighbor_geom_weights(cfg: PressureCASimConfig):
    offsets = _neighbor_offsets()
    raw = []
    dists = []
    for ox, oy in offsets:
        dist = _neighbor_distance(cfg.dx, cfg.dy, ox, oy)
        interface_len = _effective_interface_length(
            dx=cfg.dx,
            dy=cfg.dy,
            ox=ox,
            oy=oy,
            diagonal_scale=cfg.diagonal_interface_scale,
        )
        raw.append(interface_len / (dist * dist))
        dists.append(dist)
    raw = np.asarray(raw, dtype=np.float64)
    raw /= np.sum(raw)
    return offsets, raw, np.asarray(dists, dtype=np.float64)


def _source_extra_value(cfg: PressureCASimConfig, t: float) -> float:
    val = cfg.source_amplitude_pa * np.sin(2.0 * np.pi * cfg.source_frequency_hz * t + cfg.source_phase_rad)
    if cfg.overpressure_only:
        return max(0.0, val)
    return val


def _sim_dtype(cfg: PressureCASimConfig):
    return np.float32 if cfg.use_float32 else np.float64


def _neighbor_value_and_mask(arr: np.ndarray, ox: int, oy: int):
    ny, nx = arr.shape
    out = np.zeros_like(arr)
    valid = np.zeros(arr.shape, dtype=bool)

    x0 = max(0, -ox)
    x1 = min(nx, nx - ox)
    y0 = max(0, -oy)
    y1 = min(ny, ny - oy)
    if x1 <= x0 or y1 <= y0:
        return out, valid

    out[y0:y1, x0:x1] = arr[y0 + oy:y1 + oy, x0 + ox:x1 + ox]
    valid[y0:y1, x0:x1] = True
    return out, valid


def _shift_add_inplace(out, src, ox: int, oy: int):
    ny, nx = out.shape
    x0 = max(0, -ox)
    x1 = min(nx, nx - ox)
    y0 = max(0, -oy)
    y1 = min(ny, ny - oy)
    if x1 <= x0 or y1 <= y0:
        return
    out[y0 + oy:y1 + oy, x0 + ox:x1 + ox] += src[y0:y1, x0:x1]


def build_static_pressure(cfg: PressureCASimConfig) -> np.ndarray:
    """Hydrostatic pressure field p = ρgh for each row (depth increases with y)."""
    depth_centers = (np.arange(cfg.ny) + 0.5) * cfg.dy
    static_by_row = cfg.rho * cfg.g * depth_centers
    return np.repeat(static_by_row[:, None], cfg.nx, axis=1)


def build_sound_speed_grid(cfg: PressureCASimConfig) -> np.ndarray:
    """Interpolate the SSP depth/speed pairs onto a (ny, nx) grid.

    Each row gets one speed value (linearly interpolated from the SSP
    table).  The result is broadcast across columns since the SSP
    varies only with depth.
    """
    if len(cfg.ssp_depths_m) != len(cfg.ssp_speeds_mps):
        raise ValueError("ssp_depths_m and ssp_speeds_mps must have same length.")
    if len(cfg.ssp_depths_m) < 2:
        raise ValueError("Need at least two SSP points.")
    if np.any(np.diff(np.asarray(cfg.ssp_depths_m)) <= 0.0):
        raise ValueError("ssp_depths_m must be strictly increasing.")

    depth_centers = (np.arange(cfg.ny) + 0.5) * cfg.dy
    speeds = np.interp(depth_centers, np.asarray(cfg.ssp_depths_m), np.asarray(cfg.ssp_speeds_mps))
    return np.repeat(speeds[:, None], cfg.nx, axis=1)


def _precompute_transfer_tables(cfg: PressureCASimConfig, sound_speed: np.ndarray) -> Dict[str, np.ndarray]:
    offsets, geom_weights, dists = _neighbor_geom_weights(cfg)
    nbh = len(offsets)
    ny, nx = sound_speed.shape
    eps = 1e-12

    alpha = np.zeros((nbh, ny, nx), dtype=np.float64)
    tcoef = np.zeros((nbh, ny, nx), dtype=np.float64)
    rcoef = np.zeros((nbh, ny, nx), dtype=np.float64)
    raw_valid_sum = np.zeros((ny, nx), dtype=np.float64)
    raw_full_sum = np.zeros((ny, nx), dtype=np.float64)

    z_i = cfg.rho * sound_speed
    c_i = sound_speed

    for k, (ox, oy) in enumerate(offsets):
        c_j, valid = _neighbor_value_and_mask(sound_speed, ox, oy)
        z_j = cfg.rho * c_j

        c_pair = 2.0 * c_i * c_j / (c_i + c_j + eps)
        local_courant = np.clip(c_pair * cfg.dt / max(dists[k], eps), 0.0, 1.0)
        refr_gain = 1.0 + cfg.refraction_strength * ((c_j - c_i) / np.maximum(c_i, eps))
        refr_gain = np.clip(refr_gain, 0.2, 2.0)

        raw_valid = geom_weights[k] * local_courant * refr_gain * valid
        raw_valid_sum += raw_valid

        # Used only to compute boundary reflection ratio: invalid neighbors contribute to "would-be" split.
        full_courant = np.clip(c_i * cfg.dt / max(dists[k], eps), 0.0, 1.0)
        raw_full_sum += geom_weights[k] * full_courant

        denom = np.maximum((z_i + z_j) ** 2, eps)
        tcoef_k = 4.0 * z_i * z_j / denom
        rcoef_k = ((z_j - z_i) ** 2) / denom
        tcoef[k] = tcoef_k * valid
        rcoef[k] = rcoef_k * valid
        alpha[k] = raw_valid

    # Boundary reflection term: part of transferable energy whose neighbor lies outside domain.
    boundary_ratio = 1.0 - (raw_valid_sum / np.maximum(raw_full_sum, eps))
    boundary_ratio = np.clip(boundary_ratio, 0.0, 1.0)

    valid_norm = np.maximum(raw_valid_sum, eps)
    alpha = alpha / valid_norm[None, :, :]
    alpha *= (1.0 - boundary_ratio)[None, :, :]
    # Sum_k alpha_k = 1 - boundary_ratio

    return {
        "alpha": alpha.astype(np.float64),
        "tcoef": tcoef.astype(np.float64),
        "rcoef": rcoef.astype(np.float64),
        "boundary_ratio": boundary_ratio.astype(np.float64),
        "offsets": np.asarray(offsets, dtype=np.int64),
    }


def _run_numpy_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
    tables: Dict[str, np.ndarray],
):
    dtype = _sim_dtype(cfg)
    alpha = tables["alpha"].astype(dtype, copy=False)
    tcoef = tables["tcoef"].astype(dtype, copy=False)
    rcoef = tables["rcoef"].astype(dtype, copy=False)
    boundary_ratio = tables["boundary_ratio"].astype(dtype, copy=False)
    offsets = tables["offsets"]

    extra = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype)

    for step in range(cfg.steps):
        t = step * cfg.dt

        base = np.maximum(extra, 0.0) if cfg.overpressure_only else extra
        transferable = cfg.transfer_fraction * base
        next_extra = base - transferable
        next_extra += transferable * boundary_ratio

        for k in range(offsets.shape[0]):
            ox = int(offsets[k, 0])
            oy = int(offsets[k, 1])
            portion = transferable * alpha[k]
            if cfg.use_impedance_interface:
                transmitted = portion * tcoef[k]
                reflected = portion * rcoef[k]
                next_extra += reflected
                _shift_add_inplace(next_extra, transmitted, ox, oy)
            else:
                _shift_add_inplace(next_extra, portion, ox, oy)

        if cfg.damping > 0.0:
            next_extra *= (1.0 - cfg.damping)

        src_val = _source_extra_value(cfg, t)
        next_extra[cfg.source_iy, cfg.source_ix] = src_val
        source_trace[step] = src_val

        extra = next_extra
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(extra.copy())

    return {
        "backend_used": "cpu-numpy-vectorized",
        "static_pressure": static_pressure.astype(dtype, copy=False),
        "sound_speed": sound_speed.astype(dtype, copy=False),
        "final_extra_pressure": extra,
        "final_total_pressure": static_pressure.astype(dtype, copy=False) + extra,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


def _run_cupy_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
    tables: Dict[str, np.ndarray],
):
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy backend requested but cupy is not available.")
    _ = cp.cuda.runtime.getDeviceCount()

    dtype_np = _sim_dtype(cfg)
    dtype_cp = cp.float32 if dtype_np == np.float32 else cp.float64

    alpha = cp.asarray(tables["alpha"], dtype=dtype_cp)
    tcoef = cp.asarray(tables["tcoef"], dtype=dtype_cp)
    rcoef = cp.asarray(tables["rcoef"], dtype=dtype_cp)
    boundary_ratio = cp.asarray(tables["boundary_ratio"], dtype=dtype_cp)
    offsets = tables["offsets"]

    extra = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype_np)

    for step in range(cfg.steps):
        t = step * cfg.dt
        base = cp.maximum(extra, 0.0) if cfg.overpressure_only else extra
        transferable = cfg.transfer_fraction * base
        next_extra = base - transferable
        next_extra += transferable * boundary_ratio

        for k in range(offsets.shape[0]):
            ox = int(offsets[k, 0])
            oy = int(offsets[k, 1])
            portion = transferable * alpha[k]
            if cfg.use_impedance_interface:
                transmitted = portion * tcoef[k]
                reflected = portion * rcoef[k]
                next_extra += reflected
                _shift_add_inplace(next_extra, transmitted, ox, oy)
            else:
                _shift_add_inplace(next_extra, portion, ox, oy)

        if cfg.damping > 0.0:
            next_extra *= (1.0 - cfg.damping)

        src_val = _source_extra_value(cfg, t)
        next_extra[cfg.source_iy, cfg.source_ix] = dtype_cp(src_val)
        source_trace[step] = src_val

        extra = next_extra
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(cp.asnumpy(extra))

    cp.cuda.Stream.null.synchronize()
    extra_np = cp.asnumpy(extra)
    return {
        "backend_used": "gpu-cupy",
        "static_pressure": static_pressure.astype(dtype_np, copy=False),
        "sound_speed": sound_speed.astype(dtype_np, copy=False),
        "final_extra_pressure": extra_np,
        "final_total_pressure": static_pressure.astype(dtype_np, copy=False) + extra_np,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


def _check_wave_cfl(cfg: PressureCASimConfig, sound_speed: np.ndarray):
    cmax = float(np.max(sound_speed))
    cmin = float(np.min(sound_speed))
    cfl = cmax * cfg.dt * np.sqrt((1.0 / (cfg.dx * cfg.dx)) + (1.0 / (cfg.dy * cfg.dy)))
    if cfl >= 1.0:
        raise ValueError(
            f"Wave CFL unstable: {cfl:.3f} >= 1.0. Reduce dt or increase dx/dy."
        )

    f = cfg.source_frequency_hz
    if f > 0:
        # Temporal Nyquist: need at least 2 samples per cycle (preferably ≥10)
        f_nyquist = 0.5 / cfg.dt
        if f > f_nyquist:
            import warnings
            warnings.warn(
                f"Source frequency {f:.0f} Hz exceeds temporal Nyquist limit "
                f"({f_nyquist:.0f} Hz at dt={cfg.dt:.6f}s). "
                f"The source signal is aliased — lower frequency or reduce dt.",
                stacklevel=3,
            )
        elif f > f_nyquist * 0.4:
            import warnings
            warnings.warn(
                f"Source {f:.0f} Hz is near Nyquist ({f_nyquist:.0f} Hz). "
                f"Only {1.0/(f*cfg.dt):.1f} samples/cycle — consider reducing dt.",
                stacklevel=3,
            )

        # Spatial: need ≥5 cells per wavelength for reasonable accuracy
        dx_max = max(cfg.dx, cfg.dy)
        wavelength = cmin / f
        cells_per_lambda = wavelength / dx_max
        if cells_per_lambda < 2.0:
            import warnings
            warnings.warn(
                f"Wavelength {wavelength:.2f}m < 2 cells ({dx_max:.2f}m). "
                f"Grid cannot represent {f:.0f} Hz. Lower frequency or reduce dx/dy.",
                stacklevel=3,
            )
        elif cells_per_lambda < 5.0:
            import warnings
            warnings.warn(
                f"Only {cells_per_lambda:.1f} cells/wavelength for {f:.0f} Hz "
                f"(λ={wavelength:.2f}m, dx={dx_max:.2f}m). "
                f"Results may be inaccurate — prefer ≥5 cells/λ.",
                stacklevel=3,
            )


def auto_resolve_grid(
    frequency_hz: float,
    c_min: float = 1490.0,
    c_max: float = 1540.0,
    width_m: float | None = None,
    height_m: float | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
    dt: float = 0.0,
    nx: int = 0,
    ny: int = 0,
    cells_per_wavelength: int = 10,
    cfl_factor: float = 0.45,
) -> dict:
    """Compute grid/time parameters that avoid aliasing for a given frequency.

    Any parameter left at 0 (or None for sizes) is auto-computed.
    Explicitly provided non-zero values are kept as-is.

    Returns dict with keys: dx, dy, dt, nx, ny, width_m, height_m.
    """
    wl_min = c_min / max(frequency_hz, 1e-6)

    if dx <= 0:
        dx = wl_min / cells_per_wavelength
    if dy <= 0:
        dy = dx

    if dt <= 0:
        dt = cfl_factor / (c_max * ((1.0 / dx**2 + 1.0 / dy**2) ** 0.5))

    if width_m is not None and width_m > 0:
        if nx <= 0:
            nx = max(4, int(np.ceil(width_m / dx)))
    elif nx > 0:
        width_m = nx * dx
    else:
        nx = 160
        width_m = nx * dx

    if height_m is not None and height_m > 0:
        if ny <= 0:
            ny = max(4, int(np.ceil(height_m / dy)))
    elif ny > 0:
        height_m = ny * dy
    else:
        ny = 90
        height_m = ny * dy

    return dict(dx=dx, dy=dy, dt=dt, nx=nx, ny=ny,
                width_m=width_m, height_m=height_m)


def _apply_open_ranges_1d(values: np.ndarray, ranges: Tuple[Tuple[int, int], ...]):
    n = values.shape[0]
    for start, end in ranges:
        s = max(0, int(start))
        e = min(n, int(end))
        if e > s:
            values[s:e] = 0.0


def _build_boundary_reflection_profiles(cfg: PressureCASimConfig):
    top = np.full(cfg.nx, float(cfg.boundary_reflect_top), dtype=np.float64)
    bottom = np.full(cfg.nx, float(cfg.boundary_reflect_bottom), dtype=np.float64)
    left = np.full(cfg.ny, float(cfg.boundary_reflect_left), dtype=np.float64)
    right = np.full(cfg.ny, float(cfg.boundary_reflect_right), dtype=np.float64)

    _apply_open_ranges_1d(top, cfg.top_open_x_ranges)
    _apply_open_ranges_1d(bottom, cfg.bottom_open_x_ranges)
    _apply_open_ranges_1d(left, cfg.left_open_y_ranges)
    _apply_open_ranges_1d(right, cfg.right_open_y_ranges)

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }


def _run_wave_numpy_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
):
    dtype = _sim_dtype(cfg)
    p_prev = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    p = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype)

    inv_dx2 = dtype(1.0 / (cfg.dx * cfg.dx))
    inv_dy2 = dtype(1.0 / (cfg.dy * cfg.dy))
    c2dt2 = (sound_speed.astype(dtype, copy=False) * dtype(cfg.dt)) ** 2
    sigma_dt = dtype(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    boundary_profiles = _build_boundary_reflection_profiles(cfg)
    top_reflect = boundary_profiles["top"].astype(dtype, copy=False)
    bottom_reflect = boundary_profiles["bottom"].astype(dtype, copy=False)
    left_reflect = boundary_profiles["left"].astype(dtype, copy=False)
    right_reflect = boundary_profiles["right"].astype(dtype, copy=False)

    for step in range(cfg.steps):
        t = step * cfg.dt
        p_next = np.zeros_like(p)

        lap = (
            (p[1:-1, 2:] - dtype(2.0) * p[1:-1, 1:-1] + p[1:-1, :-2]) * inv_dx2
            + (p[2:, 1:-1] - dtype(2.0) * p[1:-1, 1:-1] + p[:-2, 1:-1]) * inv_dy2
        )
        p_next[1:-1, 1:-1] = (
            (dtype(2.0) - sigma_dt) * p[1:-1, 1:-1]
            - (dtype(1.0) - sigma_dt) * p_prev[1:-1, 1:-1]
            + c2dt2[1:-1, 1:-1] * lap
        )

        # Boundary reflections.
        p_next[0, :] = top_reflect * p_next[1, :]
        p_next[-1, :] = bottom_reflect * p_next[-2, :]
        p_next[:, 0] = left_reflect * p_next[:, 1]
        p_next[:, -1] = right_reflect * p_next[:, -2]

        src_val = _source_extra_value(cfg, t)
        p_next[cfg.source_iy, cfg.source_ix] += dtype(src_val)
        source_trace[step] = src_val

        p_prev, p = p, p_next
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(p.copy())

    return {
        "backend_used": "cpu-numpy-wave",
        "static_pressure": static_pressure.astype(dtype, copy=False),
        "sound_speed": sound_speed.astype(dtype, copy=False),
        "final_extra_pressure": p,
        "final_total_pressure": static_pressure.astype(dtype, copy=False) + p,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


# ─── Numba parallel CPU backend ────────────────────────────────────────────

def _build_numba_wave_kernel():
    """JIT-compile the fused wave step at first call; cached thereafter."""
    if not NUMBA_AVAILABLE:
        return None

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _step(p, p_prev, p_next, c2dt2, inv_dx2, inv_dy2, sigma_dt,
              top_r, bot_r, left_r, right_r):
        ny = p.shape[0]
        nx = p.shape[1]
        two_m_s = 2.0 - sigma_dt
        one_m_s = 1.0 - sigma_dt
        for j in nb.prange(1, ny - 1):
            for i in range(1, nx - 1):
                center = p[j, i]
                lap = ((p[j, i + 1] - 2.0 * center + p[j, i - 1]) * inv_dx2
                       + (p[j + 1, i] - 2.0 * center + p[j - 1, i]) * inv_dy2)
                p_next[j, i] = (two_m_s * center
                                - one_m_s * p_prev[j, i]
                                + c2dt2[j, i] * lap)
        for i in range(nx):
            p_next[0, i] = top_r[i] * p_next[1, i]
            p_next[ny - 1, i] = bot_r[i] * p_next[ny - 2, i]
        for j in range(ny):
            p_next[j, 0] = left_r[j] * p_next[j, 1]
            p_next[j, nx - 1] = right_r[j] * p_next[j, nx - 2]

    return _step


_numba_wave_step = None


def _get_numba_wave_step():
    global _numba_wave_step
    if _numba_wave_step is None:
        _numba_wave_step = _build_numba_wave_kernel()
    return _numba_wave_step


def _run_wave_numba_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba not available.")
    kernel = _get_numba_wave_step()
    if kernel is None:
        raise RuntimeError("Failed to compile Numba kernel.")

    dtype = _sim_dtype(cfg)
    p_prev = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    p = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    p_next = np.zeros((cfg.ny, cfg.nx), dtype=dtype)
    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype)

    inv_dx2 = dtype(1.0 / (cfg.dx * cfg.dx))
    inv_dy2 = dtype(1.0 / (cfg.dy * cfg.dy))
    c2dt2 = (sound_speed.astype(dtype, copy=False) * dtype(cfg.dt)) ** 2
    sigma_dt = dtype(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    bnd = _build_boundary_reflection_profiles(cfg)
    top_r = bnd["top"].astype(dtype, copy=False)
    bot_r = bnd["bottom"].astype(dtype, copy=False)
    left_r = bnd["left"].astype(dtype, copy=False)
    right_r = bnd["right"].astype(dtype, copy=False)

    # Warm up JIT (first call compiles; subsequent calls use cache).
    kernel(p, p_prev, p_next, c2dt2, inv_dx2, inv_dy2, sigma_dt,
           top_r, bot_r, left_r, right_r)
    p.fill(0); p_prev.fill(0); p_next.fill(0)

    for step in range(cfg.steps):
        t = step * cfg.dt
        kernel(p, p_prev, p_next, c2dt2, inv_dx2, inv_dy2, sigma_dt,
               top_r, bot_r, left_r, right_r)

        src_val = dtype(_source_extra_value(cfg, t))
        p_next[cfg.source_iy, cfg.source_ix] += src_val
        source_trace[step] = src_val

        p_prev, p, p_next = p, p_next, p_prev
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(p.copy())

    n_threads = nb.config.NUMBA_NUM_THREADS
    return {
        "backend_used": f"cpu-numba-wave-{n_threads}threads",
        "static_pressure": static_pressure.astype(dtype, copy=False),
        "sound_speed": sound_speed.astype(dtype, copy=False),
        "final_extra_pressure": p,
        "final_total_pressure": static_pressure.astype(dtype, copy=False) + p,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


# ─── CuPy fused CUDA kernel backend ────────────────────────────────────────

_WAVE_CUDA_SRC_F32 = r'''
extern "C" __global__
void wave_step_f32(
    const float* __restrict__ p,
    const float* __restrict__ p_prev,
    const float* __restrict__ c2dt2,
    float* __restrict__ p_next,
    const int ny, const int nx,
    const float inv_dx2, const float inv_dy2,
    const float two_m_s, const float one_m_s
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ny * nx) return;
    int j = idx / nx;
    int i = idx % nx;
    if (j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
        float c = p[idx];
        float lap = (p[idx+1] - 2.0f*c + p[idx-1]) * inv_dx2
                  + (p[idx+nx] - 2.0f*c + p[idx-nx]) * inv_dy2;
        p_next[idx] = two_m_s*c - one_m_s*p_prev[idx] + c2dt2[idx]*lap;
    }
}
'''

_WAVE_CUDA_SRC_F64 = r'''
extern "C" __global__
void wave_step_f64(
    const double* __restrict__ p,
    const double* __restrict__ p_prev,
    const double* __restrict__ c2dt2,
    double* __restrict__ p_next,
    const int ny, const int nx,
    const double inv_dx2, const double inv_dy2,
    const double two_m_s, const double one_m_s
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= ny * nx) return;
    int j = idx / nx;
    int i = idx % nx;
    if (j >= 1 && j < ny - 1 && i >= 1 && i < nx - 1) {
        double c = p[idx];
        double lap = (p[idx+1] - 2.0*c + p[idx-1]) * inv_dx2
                   + (p[idx+nx] - 2.0*c + p[idx-nx]) * inv_dy2;
        p_next[idx] = two_m_s*c - one_m_s*p_prev[idx] + c2dt2[idx]*lap;
    }
}
'''

# Boundary + source kernel: applied AFTER the interior update kernel.
# Runs on boundary cells only (max 2*(nx+ny) threads) -- very lightweight.
_WAVE_BOUNDARY_CUDA_TEMPLATE = r'''
extern "C" __global__
void wave_boundary_{T}(
    {F}* __restrict__ p_next,
    const int ny, const int nx,
    const {F}* __restrict__ top_r,
    const {F}* __restrict__ bot_r,
    const {F}* __restrict__ left_r,
    const {F}* __restrict__ right_r,
    const int src_iy, const int src_ix,
    const {F} src_val
) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int perimeter = 2 * nx + 2 * (ny - 2);
    if (idx >= perimeter) return;

    if (idx < nx) {{
        // Top row: boundary cell (0, i) = top_r[i] * p_next(1, i)
        int i = idx;
        p_next[i] = top_r[i] * p_next[nx + i];
    }} else if (idx < 2 * nx) {{
        // Bottom row
        int i = idx - nx;
        p_next[(ny - 1) * nx + i] = bot_r[i] * p_next[(ny - 2) * nx + i];
    }} else if (idx < 2 * nx + (ny - 2)) {{
        // Left col (excluding corners)
        int j = idx - 2 * nx + 1;
        p_next[j * nx] = left_r[j] * p_next[j * nx + 1];
    }} else {{
        // Right col (excluding corners)
        int j = idx - 2 * nx - (ny - 2) + 1;
        p_next[j * nx + nx - 1] = right_r[j] * p_next[j * nx + nx - 2];
    }}

    // Source injection (only one thread matches)
    if (idx == 0) {{
        p_next[src_iy * nx + src_ix] += src_val;
    }}
}}
'''

_cuda_kernels: dict = {}


def _get_cuda_wave_kernel(use_float32: bool):
    key = "f32" if use_float32 else "f64"
    if key not in _cuda_kernels:
        if use_float32:
            _cuda_kernels[key] = cp.RawKernel(_WAVE_CUDA_SRC_F32, "wave_step_f32")
        else:
            _cuda_kernels[key] = cp.RawKernel(_WAVE_CUDA_SRC_F64, "wave_step_f64")
    return _cuda_kernels[key]


def _get_cuda_boundary_kernel(use_float32: bool):
    key = "bnd_f32" if use_float32 else "bnd_f64"
    if key not in _cuda_kernels:
        T = "f32" if use_float32 else "f64"
        F = "float" if use_float32 else "double"
        src = _WAVE_BOUNDARY_CUDA_TEMPLATE.format(T=T, F=F)
        _cuda_kernels[key] = cp.RawKernel(src, f"wave_boundary_{T}")
    return _cuda_kernels[key]


def _run_wave_cupy_fused_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
):
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy backend requested but cupy is not available.")
    _ = cp.cuda.runtime.getDeviceCount()

    dtype_np = _sim_dtype(cfg)
    dtype_cp = cp.float32 if dtype_np == np.float32 else cp.float64
    kernel = _get_cuda_wave_kernel(cfg.use_float32)

    total_cells = cfg.ny * cfg.nx
    block = 256
    grid = (total_cells + block - 1) // block

    p_prev = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    p = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    p_next = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    c2dt2 = (cp.asarray(sound_speed, dtype=dtype_cp) * dtype_cp(cfg.dt)) ** 2

    inv_dx2 = dtype_np(1.0 / (cfg.dx * cfg.dx))
    inv_dy2 = dtype_np(1.0 / (cfg.dy * cfg.dy))
    sigma_dt = float(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    two_m_s = dtype_np(2.0 - sigma_dt)
    one_m_s = dtype_np(1.0 - sigma_dt)

    bnd = _build_boundary_reflection_profiles(cfg)
    top_r = cp.asarray(bnd["top"], dtype=dtype_cp)
    bot_r = cp.asarray(bnd["bottom"], dtype=dtype_cp)
    left_r = cp.asarray(bnd["left"], dtype=dtype_cp)
    right_r = cp.asarray(bnd["right"], dtype=dtype_cp)

    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype_np)

    ny_i = np.int32(cfg.ny)
    nx_i = np.int32(cfg.nx)

    for step in range(cfg.steps):
        t = step * cfg.dt

        kernel((grid,), (block,),
               (p, p_prev, c2dt2, p_next,
                ny_i, nx_i, inv_dx2, inv_dy2, two_m_s, one_m_s))

        p_next[0, :] = top_r * p_next[1, :]
        p_next[-1, :] = bot_r * p_next[-2, :]
        p_next[:, 0] = left_r * p_next[:, 1]
        p_next[:, -1] = right_r * p_next[:, -2]

        src_val = _source_extra_value(cfg, t)
        p_next[cfg.source_iy, cfg.source_ix] += dtype_cp(src_val)
        source_trace[step] = src_val

        p_prev, p, p_next = p, p_next, p_prev
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(cp.asnumpy(p))

    cp.cuda.Stream.null.synchronize()
    p_np = cp.asnumpy(p)
    dev = cp.cuda.Device(0)
    dev_name = dev.attributes.get("DeviceName", f"GPU-{dev.id}")
    return {
        "backend_used": f"gpu-cupy-fused-wave ({dev_name})",
        "static_pressure": static_pressure.astype(dtype_np, copy=False),
        "sound_speed": sound_speed.astype(dtype_np, copy=False),
        "final_extra_pressure": p_np,
        "final_total_pressure": static_pressure.astype(dtype_np, copy=False) + p_np,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


def _run_wave_cupy_backend(
    cfg: PressureCASimConfig,
    static_pressure: np.ndarray,
    sound_speed: np.ndarray,
):
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy backend requested but cupy is not available.")
    _ = cp.cuda.runtime.getDeviceCount()

    dtype_np = _sim_dtype(cfg)
    dtype_cp = cp.float32 if dtype_np == np.float32 else cp.float64

    p_prev = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    p = cp.zeros((cfg.ny, cfg.nx), dtype=dtype_cp)
    c2dt2 = (cp.asarray(sound_speed, dtype=dtype_cp) * dtype_cp(cfg.dt)) ** 2
    inv_dx2 = dtype_cp(1.0 / (cfg.dx * cfg.dx))
    inv_dy2 = dtype_cp(1.0 / (cfg.dy * cfg.dy))
    sigma_dt = dtype_cp(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    boundary_profiles = _build_boundary_reflection_profiles(cfg)
    top_reflect = cp.asarray(boundary_profiles["top"], dtype=dtype_cp)
    bottom_reflect = cp.asarray(boundary_profiles["bottom"], dtype=dtype_cp)
    left_reflect = cp.asarray(boundary_profiles["left"], dtype=dtype_cp)
    right_reflect = cp.asarray(boundary_profiles["right"], dtype=dtype_cp)

    frames_extra: List[np.ndarray] = []
    source_trace = np.zeros(cfg.steps, dtype=dtype_np)

    for step in range(cfg.steps):
        t = step * cfg.dt
        p_next = cp.zeros_like(p)

        lap = (
            (p[1:-1, 2:] - dtype_cp(2.0) * p[1:-1, 1:-1] + p[1:-1, :-2]) * inv_dx2
            + (p[2:, 1:-1] - dtype_cp(2.0) * p[1:-1, 1:-1] + p[:-2, 1:-1]) * inv_dy2
        )
        p_next[1:-1, 1:-1] = (
            (dtype_cp(2.0) - sigma_dt) * p[1:-1, 1:-1]
            - (dtype_cp(1.0) - sigma_dt) * p_prev[1:-1, 1:-1]
            + c2dt2[1:-1, 1:-1] * lap
        )

        p_next[0, :] = top_reflect * p_next[1, :]
        p_next[-1, :] = bottom_reflect * p_next[-2, :]
        p_next[:, 0] = left_reflect * p_next[:, 1]
        p_next[:, -1] = right_reflect * p_next[:, -2]

        src_val = _source_extra_value(cfg, t)
        p_next[cfg.source_iy, cfg.source_ix] += dtype_cp(src_val)
        source_trace[step] = src_val

        p_prev, p = p, p_next
        if step % max(1, cfg.frame_stride) == 0:
            frames_extra.append(cp.asnumpy(p))

    cp.cuda.Stream.null.synchronize()
    p_np = cp.asnumpy(p)
    return {
        "backend_used": "gpu-cupy-wave",
        "static_pressure": static_pressure.astype(dtype_np, copy=False),
        "sound_speed": sound_speed.astype(dtype_np, copy=False),
        "final_extra_pressure": p_np,
        "final_total_pressure": static_pressure.astype(dtype_np, copy=False) + p_np,
        "frames_extra": frames_extra,
        "source_trace_extra": source_trace,
    }


# ─── Single-step interface for interactive use ─────────────────────────────

class WaveStepper:
    """Wraps wave simulation state for single-step advancing.

    Picks the best available backend (GPU > Numba > NumPy) and exposes
    a ``step()`` method that advances one timestep and returns the current
    pressure field as a numpy array.
    """

    def __init__(self, cfg: PressureCASimConfig):
        if cfg.nx <= 2 or cfg.ny <= 2:
            raise ValueError("Grid must be at least 3x3.")
        if cfg.dt <= 0:
            raise ValueError("dt must be > 0.")
        if not (0 <= cfg.source_ix < cfg.nx and 0 <= cfg.source_iy < cfg.ny):
            raise ValueError("Source index is outside the grid.")

        self.cfg = cfg
        self._step_count = 0
        self._dtype = _sim_dtype(cfg)
        self._sound_speed = build_sound_speed_grid(cfg)
        _check_wave_cfl(cfg, self._sound_speed)

        bnd = _build_boundary_reflection_profiles(cfg)

        requested = str(cfg.backend).strip().lower()
        self._backend_name = "unknown"
        self._use_gpu = False

        # Try GPU first
        if requested in ("auto", "gpu") and CUPY_AVAILABLE:
            try:
                self._init_gpu(bnd)
                return
            except Exception:
                if requested == "gpu":
                    raise

        # Try Numba
        if requested in ("auto", "cpu") and NUMBA_AVAILABLE:
            try:
                self._init_numba(bnd)
                return
            except Exception:
                pass

        # Fallback: NumPy
        self._init_numpy(bnd)

    def _init_numpy(self, bnd):
        dt = self._dtype
        self._p_prev = np.zeros((self.cfg.ny, self.cfg.nx), dtype=dt)
        self._p = np.zeros((self.cfg.ny, self.cfg.nx), dtype=dt)
        self._inv_dx2 = dt(1.0 / (self.cfg.dx ** 2))
        self._inv_dy2 = dt(1.0 / (self.cfg.dy ** 2))
        self._c2dt2 = (self._sound_speed.astype(dt, copy=False) * dt(self.cfg.dt)) ** 2
        self._sigma_dt = dt(max(0.0, self.cfg.wave_absorption_per_s * self.cfg.dt))
        self._top_r = bnd["top"].astype(dt, copy=False)
        self._bot_r = bnd["bottom"].astype(dt, copy=False)
        self._left_r = bnd["left"].astype(dt, copy=False)
        self._right_r = bnd["right"].astype(dt, copy=False)
        self._backend_name = "cpu-numpy-wave"
        self._step_fn = self._step_numpy

    def _init_numba(self, bnd):
        kernel = _get_numba_wave_step()
        if kernel is None:
            raise RuntimeError("Numba kernel compile failed")
        self._init_numpy(bnd)
        self._p_next_buf = np.zeros_like(self._p)
        self._nb_kernel = kernel
        # Warm up JIT
        self._nb_kernel(self._p, self._p_prev, self._p_next_buf,
                        self._c2dt2, self._inv_dx2, self._inv_dy2, self._sigma_dt,
                        self._top_r, self._bot_r, self._left_r, self._right_r)
        self._p.fill(0); self._p_prev.fill(0); self._p_next_buf.fill(0)
        n_threads = nb.config.NUMBA_NUM_THREADS
        self._backend_name = f"cpu-numba-wave-{n_threads}threads"
        self._step_fn = self._step_numba

    def _init_gpu(self, bnd):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        _ = cp.cuda.runtime.getDeviceCount()
        dt_np = self._dtype
        dt_cp = cp.float32 if dt_np == np.float32 else cp.float64
        self._dt_cp = dt_cp
        self._gpu_kernel = _get_cuda_wave_kernel(self.cfg.use_float32)
        self._gpu_bnd_kernel = _get_cuda_boundary_kernel(self.cfg.use_float32)
        ny, nx = self.cfg.ny, self.cfg.nx
        self._gpu_p_prev = cp.zeros((ny, nx), dtype=dt_cp)
        self._gpu_p = cp.zeros((ny, nx), dtype=dt_cp)
        self._gpu_p_next = cp.zeros((ny, nx), dtype=dt_cp)
        self._gpu_c2dt2 = (cp.asarray(self._sound_speed, dtype=dt_cp) * dt_cp(self.cfg.dt)) ** 2
        sigma_dt = float(max(0.0, self.cfg.wave_absorption_per_s * self.cfg.dt))
        self._gpu_two_m_s = dt_np(2.0 - sigma_dt)
        self._gpu_one_m_s = dt_np(1.0 - sigma_dt)
        self._gpu_inv_dx2 = dt_np(1.0 / (self.cfg.dx ** 2))
        self._gpu_inv_dy2 = dt_np(1.0 / (self.cfg.dy ** 2))
        self._gpu_ny = np.int32(ny)
        self._gpu_nx = np.int32(nx)
        total = ny * nx
        self._gpu_block = 256
        self._gpu_grid = (total + self._gpu_block - 1) // self._gpu_block
        perimeter = 2 * nx + 2 * (ny - 2)
        self._gpu_bnd_block = min(256, perimeter)
        self._gpu_bnd_grid = (perimeter + self._gpu_bnd_block - 1) // self._gpu_bnd_block
        self._gpu_top_r = cp.asarray(bnd["top"], dtype=dt_cp)
        self._gpu_bot_r = cp.asarray(bnd["bottom"], dtype=dt_cp)
        self._gpu_left_r = cp.asarray(bnd["left"], dtype=dt_cp)
        self._gpu_right_r = cp.asarray(bnd["right"], dtype=dt_cp)
        dev = cp.cuda.Device(0)
        dev_name = dev.attributes.get("DeviceName", f"GPU-{dev.id}")
        self._backend_name = f"gpu-cupy-fused ({dev_name})"
        self._use_gpu = True
        self._step_fn = self._step_gpu

    def _step_numpy(self):
        cfg = self.cfg
        dt = self._dtype
        t = self._step_count * cfg.dt
        p, p_prev = self._p, self._p_prev

        p_next = np.zeros_like(p)
        lap = (
            (p[1:-1, 2:] - dt(2.0) * p[1:-1, 1:-1] + p[1:-1, :-2]) * self._inv_dx2
            + (p[2:, 1:-1] - dt(2.0) * p[1:-1, 1:-1] + p[:-2, 1:-1]) * self._inv_dy2
        )
        p_next[1:-1, 1:-1] = (
            (dt(2.0) - self._sigma_dt) * p[1:-1, 1:-1]
            - (dt(1.0) - self._sigma_dt) * p_prev[1:-1, 1:-1]
            + self._c2dt2[1:-1, 1:-1] * lap
        )
        p_next[0, :] = self._top_r * p_next[1, :]
        p_next[-1, :] = self._bot_r * p_next[-2, :]
        p_next[:, 0] = self._left_r * p_next[:, 1]
        p_next[:, -1] = self._right_r * p_next[:, -2]

        src_val = _source_extra_value(cfg, t)
        p_next[cfg.source_iy, cfg.source_ix] += dt(src_val)

        self._p_prev = p
        self._p = p_next
        return src_val

    def _step_numba(self):
        cfg = self.cfg
        t = self._step_count * cfg.dt

        self._nb_kernel(self._p, self._p_prev, self._p_next_buf,
                        self._c2dt2, self._inv_dx2, self._inv_dy2, self._sigma_dt,
                        self._top_r, self._bot_r, self._left_r, self._right_r)

        src_val = self._dtype(_source_extra_value(cfg, t))
        self._p_next_buf[cfg.source_iy, cfg.source_ix] += src_val

        self._p_prev, self._p, self._p_next_buf = self._p, self._p_next_buf, self._p_prev
        return float(src_val)

    def _step_gpu(self):
        cfg = self.cfg
        t = self._step_count * cfg.dt
        src_val = _source_extra_value(cfg, t)

        self._gpu_kernel(
            (self._gpu_grid,), (self._gpu_block,),
            (self._gpu_p, self._gpu_p_prev, self._gpu_c2dt2, self._gpu_p_next,
             self._gpu_ny, self._gpu_nx,
             self._gpu_inv_dx2, self._gpu_inv_dy2,
             self._gpu_two_m_s, self._gpu_one_m_s))
        self._gpu_bnd_kernel(
            (self._gpu_bnd_grid,), (self._gpu_bnd_block,),
            (self._gpu_p_next,
             self._gpu_ny, self._gpu_nx,
             self._gpu_top_r, self._gpu_bot_r,
             self._gpu_left_r, self._gpu_right_r,
             np.int32(cfg.source_iy), np.int32(cfg.source_ix),
             self._dt_cp(src_val)))

        self._gpu_p_prev, self._gpu_p, self._gpu_p_next = (
            self._gpu_p, self._gpu_p_next, self._gpu_p_prev)
        return float(src_val)

    def step(self) -> np.ndarray:
        """Advance one timestep. Returns current pressure field (numpy)."""
        self._step_fn()
        self._step_count += 1
        return self.pressure

    def step_n(self, n: int,
               probe_cells: list[tuple[int, int]] | None = None,
               probe_stride: int = 1,
               ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Advance *n* timesteps in a tight loop, minimizing GPU→CPU transfers.

        Parameters
        ----------
        n : int
            Number of timesteps to advance.
        probe_cells : list of (ix, iy) or None
            Grid cells to record pressure at.
        probe_stride : int
            Sample probes every *probe_stride* steps (default 1 = every step).

        Returns
        -------
        field : np.ndarray (ny, nx)
            Pressure field after the last step.
        probe_values : np.ndarray (n_samples, n_probes) float64 or None
            Pressure at each probe cell at sampled steps.
        probe_times : np.ndarray (n_samples,) float64 or None
            Simulation time at each sampled step.
        """
        cfg = self.cfg
        n_probes = len(probe_cells) if probe_cells else 0
        probe_stride = max(1, probe_stride)

        if n_probes:
            n_samples = (n + probe_stride - 1) // probe_stride
            probe_vals = np.empty((n_samples, n_probes), dtype=np.float64)
            probe_ts = np.empty(n_samples, dtype=np.float64)
        else:
            n_samples = 0
            probe_vals = None
            probe_ts = None

        if self._use_gpu:
            self._step_n_gpu(n, probe_cells, probe_vals, probe_ts, probe_stride)
        elif hasattr(self, "_nb_kernel"):
            self._step_n_numba(n, probe_cells, probe_vals, probe_ts, probe_stride)
        else:
            self._step_n_numpy(n, probe_cells, probe_vals, probe_ts, probe_stride)

        return self.pressure, probe_vals, probe_ts

    def _step_n_gpu(self, n, probe_cells, probe_vals, probe_ts, stride):
        cfg = self.cfg
        interior_kernel = self._gpu_kernel
        bnd_kernel = self._gpu_bnd_kernel
        igrid, iblock = (self._gpu_grid,), (self._gpu_block,)
        bgrid, bblock = (self._gpu_bnd_grid,), (self._gpu_bnd_block,)
        top_r, bot_r = self._gpu_top_r, self._gpu_bot_r
        left_r, right_r = self._gpu_left_r, self._gpu_right_r
        dt_cast = self._dt_cp
        p, pp, pn = self._gpu_p, self._gpu_p_prev, self._gpu_p_next
        c2dt2 = self._gpu_c2dt2
        gny, gnx = self._gpu_ny, self._gpu_nx
        inv_dx2, inv_dy2 = self._gpu_inv_dx2, self._gpu_inv_dy2
        two_m_s, one_m_s = self._gpu_two_m_s, self._gpu_one_m_s
        src_iy = np.int32(cfg.source_iy)
        src_ix = np.int32(cfg.source_ix)
        has_probes = probe_vals is not None

        # Pre-compute ALL source values on CPU in one vectorized call
        step_indices = np.arange(self._step_count, self._step_count + n)
        t_arr = step_indices * cfg.dt
        src_all = cfg.source_amplitude_pa * np.sin(
            2.0 * np.pi * cfg.source_frequency_hz * t_arr + cfg.source_phase_rad)
        if cfg.overpressure_only:
            np.maximum(src_all, 0.0, out=src_all)

        if has_probes:
            iy_arr = cp.array([iy for ix, iy in probe_cells], dtype=cp.int64)
            ix_arr = cp.array([ix for ix, iy in probe_cells], dtype=cp.int64)
            n_samples = probe_vals.shape[0]
            gpu_probe_buf = cp.empty((n_samples, len(probe_cells)), dtype=cp.float64)

        sample_idx = 0
        for i in range(n):
            src_val = dt_cast(float(src_all[i]))

            interior_kernel(igrid, iblock,
                            (p, pp, c2dt2, pn,
                             gny, gnx, inv_dx2, inv_dy2,
                             two_m_s, one_m_s))
            bnd_kernel(bgrid, bblock,
                       (pn, gny, gnx,
                        top_r, bot_r, left_r, right_r,
                        src_iy, src_ix, src_val))

            pp, p, pn = p, pn, pp
            self._step_count += 1

            if has_probes and (i % stride == stride - 1 or i == n - 1):
                gpu_probe_buf[sample_idx, :] = p[iy_arr, ix_arr]
                sample_idx += 1

        self._gpu_p_prev, self._gpu_p, self._gpu_p_next = pp, p, pn
        cp.cuda.Stream.null.synchronize()

        if has_probes:
            probe_vals[:sample_idx] = cp.asnumpy(gpu_probe_buf[:sample_idx])
            dt_val = cfg.dt
            base_step = self._step_count - n
            si = 0
            for i in range(n):
                if i % stride == stride - 1 or i == n - 1:
                    probe_ts[si] = (base_step + i + 1) * dt_val
                    si += 1

    def _step_n_numba(self, n, probe_cells, probe_vals, probe_ts, stride):
        cfg = self.cfg
        kernel = self._nb_kernel
        has_probes = probe_vals is not None
        sample_idx = 0
        for i in range(n):
            t = self._step_count * cfg.dt
            kernel(self._p, self._p_prev, self._p_next_buf,
                   self._c2dt2, self._inv_dx2, self._inv_dy2, self._sigma_dt,
                   self._top_r, self._bot_r, self._left_r, self._right_r)
            src_val = self._dtype(_source_extra_value(cfg, t))
            self._p_next_buf[cfg.source_iy, cfg.source_ix] += src_val
            self._p_prev, self._p, self._p_next_buf = (
                self._p, self._p_next_buf, self._p_prev)
            self._step_count += 1
            if has_probes and (i % stride == stride - 1 or i == n - 1):
                probe_ts[sample_idx] = self._step_count * cfg.dt
                for j, (ix, iy) in enumerate(probe_cells):
                    probe_vals[sample_idx, j] = self._p[iy, ix]
                sample_idx += 1

    def _step_n_numpy(self, n, probe_cells, probe_vals, probe_ts, stride):
        cfg = self.cfg
        has_probes = probe_vals is not None
        sample_idx = 0
        for i in range(n):
            self._step_numpy()
            self._step_count += 1
            if has_probes and (i % stride == stride - 1 or i == n - 1):
                probe_ts[sample_idx] = self._step_count * cfg.dt
                for j, (ix, iy) in enumerate(probe_cells):
                    probe_vals[sample_idx, j] = self._p[iy, ix]
                sample_idx += 1

    @property
    def pressure(self) -> np.ndarray:
        """Current pressure field as numpy array."""
        if self._use_gpu:
            return cp.asnumpy(self._gpu_p)
        return self._p.copy()

    @property
    def current_step(self) -> int:
        return self._step_count

    @property
    def current_time(self) -> float:
        return self._step_count * self.cfg.dt

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def reset(self):
        """Reset simulation to step 0."""
        self._step_count = 0
        if self._use_gpu:
            self._gpu_p.fill(0)
            self._gpu_p_prev.fill(0)
            self._gpu_p_next.fill(0)
        else:
            self._p.fill(0)
            self._p_prev.fill(0)
            if hasattr(self, "_p_next_buf"):
                self._p_next_buf.fill(0)


def run_pressure_transfer_ca(cfg: PressureCASimConfig):
    """Run the full simulation and return results as a dict.

    This is the batch entry point used by ``run_experiment.py``.
    It runs all ``cfg.steps`` timesteps, collects sampled frames,
    and returns everything needed for plotting and GIF generation.

    Returns a dict with keys:
        backend_used, static_pressure, sound_speed,
        final_extra_pressure, final_total_pressure,
        frames_extra, source_trace_extra.
    """
    if cfg.nx <= 2 or cfg.ny <= 2:
        raise ValueError("Grid must be at least 3x3.")
    if not (0.0 <= cfg.transfer_fraction <= 1.0):
        raise ValueError("transfer_fraction must be in [0, 1].")
    if not (0.0 <= cfg.damping < 1.0):
        raise ValueError("damping must be in [0, 1).")
    if cfg.dt <= 0:
        raise ValueError("dt must be > 0.")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0.")
    if not (0 <= cfg.source_ix < cfg.nx and 0 <= cfg.source_iy < cfg.ny):
        raise ValueError("Source index is outside the grid.")

    static_pressure = build_static_pressure(cfg)
    sound_speed = build_sound_speed_grid(cfg)

    requested = str(cfg.backend).strip().lower()
    if requested not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be one of: auto, cpu, gpu")
    model = str(cfg.propagation_model).strip().lower()
    if model not in ("wave", "transfer"):
        raise ValueError("propagation_model must be one of: wave, transfer")

    if model == "wave":
        _check_wave_cfl(cfg, sound_speed)
        # Priority: GPU (fused CUDA) → GPU (CuPy array) → Numba CPU → NumPy CPU.
        if requested in ("auto", "gpu"):
            try:
                result = _run_wave_cupy_fused_backend(cfg, static_pressure, sound_speed)
                result["cfg"] = cfg
                return result
            except Exception:
                pass
            try:
                result = _run_wave_cupy_backend(cfg, static_pressure, sound_speed)
                result["cfg"] = cfg
                return result
            except Exception:
                if requested == "gpu":
                    raise
        if requested in ("auto", "cpu"):
            if NUMBA_AVAILABLE:
                try:
                    result = _run_wave_numba_backend(cfg, static_pressure, sound_speed)
                    result["cfg"] = cfg
                    return result
                except Exception:
                    pass
        result = _run_wave_numpy_backend(cfg, static_pressure, sound_speed)
        result["cfg"] = cfg
        return result

    tables = _precompute_transfer_tables(cfg, sound_speed)
    if requested in ("auto", "gpu"):
        try:
            result = _run_cupy_backend(cfg, static_pressure, sound_speed, tables)
            result["cfg"] = cfg
            return result
        except Exception:
            if requested == "gpu":
                raise
    result = _run_numpy_backend(cfg, static_pressure, sound_speed, tables)
    result["cfg"] = cfg
    return result


def _pick_grid_stride(n_cells: int, approx_pixels: float,
                      min_px_per_line: float = 8.0) -> int:
    """Choose a grid-line stride so that lines are at least *min_px_per_line* apart.

    Returns 1 for cell-level grid, or a round multiple (5, 10, 20, 50, ...)
    when cells are too dense to render cleanly.
    """
    px_per_cell = approx_pixels / max(n_cells, 1)
    if px_per_cell >= min_px_per_line:
        return 1
    for stride in (2, 5, 10, 20, 50, 100):
        if px_per_cell * stride >= min_px_per_line:
            return stride
    return n_cells  # fall back to domain-edge only


def _apply_cell_grid(ax, cfg: PressureCASimConfig, color: str = "k", alpha: float = 0.25,
                      linewidth: float = 0.3, fig_width_px: float = 750.0,
                      fig_height_px: float = 450.0):
    """Draw cell-boundary grid lines, adapting stride to avoid moire aliasing."""
    sx = _pick_grid_stride(cfg.nx, fig_width_px)
    sy = _pick_grid_stride(cfg.ny, fig_height_px)

    x_edges = np.arange(0, cfg.nx + 1, sx) * cfg.dx
    y_edges = np.arange(0, cfg.ny + 1, sy) * cfg.dy

    ax.set_xticks(x_edges, minor=True)
    ax.set_yticks(y_edges, minor=True)
    ax.grid(which="minor", color=color, alpha=alpha, linewidth=linewidth)
    ax.tick_params(which="minor", length=0)

    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("k")


def plot_static_and_final(result, show_grid: bool = True):
    """Create a four-panel summary figure from simulation results.

    Panels: (1) hydrostatic pressure, (2) sound speed profile,
    (3) final acoustic pressure field, (4) source waveform.
    Returns the matplotlib Figure for saving.
    """
    cfg: PressureCASimConfig = result["cfg"]
    static_pressure = result["static_pressure"]
    sound_speed = result["sound_speed"]
    final_extra = result["final_extra_pressure"]
    final_total = result["final_total_pressure"]

    extent = [0.0, cfg.nx * cfg.dx, cfg.ny * cfg.dy, 0.0]
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)

    im0 = axes[0].imshow(static_pressure, cmap="viridis", aspect="auto", extent=extent,
                          interpolation="nearest")
    axes[0].set_title("Static Pressure (rho*g*h)")
    axes[0].set_xlabel("x distance (m)")
    axes[0].set_ylabel("depth y (m)")
    fig.colorbar(im0, ax=axes[0], label="Pa")

    im1 = axes[1].imshow(sound_speed, cmap="plasma", aspect="auto", extent=extent,
                          interpolation="nearest")
    axes[1].set_title("Sound Speed c(y)")
    axes[1].set_xlabel("x distance (m)")
    axes[1].set_ylabel("depth y (m)")
    fig.colorbar(im1, ax=axes[1], label="m/s")

    im2 = axes[2].imshow(final_extra, cmap="coolwarm", aspect="auto", extent=extent,
                          interpolation="nearest")
    axes[2].set_title("Final Extra Pressure")
    axes[2].set_xlabel("x distance (m)")
    axes[2].set_ylabel("depth y (m)")
    fig.colorbar(im2, ax=axes[2], label="Pa")

    im3 = axes[3].imshow(final_total, cmap="viridis", aspect="auto", extent=extent,
                          interpolation="nearest")
    axes[3].set_title("Final Total Pressure")
    axes[3].set_xlabel("x distance (m)")
    axes[3].set_ylabel("depth y (m)")
    fig.colorbar(im3, ax=axes[3], label="Pa")

    for ax in axes:
        ax.scatter([(cfg.source_ix + 0.5) * cfg.dx], [(cfg.source_iy + 0.5) * cfg.dy], c="white", s=14)
        if show_grid:
            _apply_cell_grid(ax, cfg)

    return fig


def animate_extra_pressure(result, fps: int = 24, show_time: bool = True,
                           show_grid: bool = True):
    """Create a matplotlib Animation of the acoustic pressure propagation."""
    cfg: PressureCASimConfig = result["cfg"]
    frames = result["frames_extra"]

    extent = [0.0, cfg.nx * cfg.dx, cfg.ny * cfg.dy, 0.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)

    vmax = np.max(np.abs(np.array(frames))) if frames else 1.0
    vmax = max(vmax, 1e-12)

    img = ax.imshow(
        frames[0],
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    ax.set_title("Extra Pressure Propagation")
    ax.set_xlabel("x distance (m)")
    ax.set_ylabel("depth y (m)")
    ax.scatter([(cfg.source_ix + 0.5) * cfg.dx], [(cfg.source_iy + 0.5) * cfg.dy], c="black", s=14)
    fig.colorbar(img, ax=ax, label="Pa")

    if show_grid:
        _apply_cell_grid(ax, cfg)

    dt_frame = cfg.dt * max(1, int(cfg.frame_stride))
    time_text = ax.text(
        0.02,
        0.96,
        "",
        transform=ax.transAxes,
        color="black",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )

    def _update(frame_idx):
        img.set_array(frames[frame_idx])
        if show_time:
            time_text.set_text(f"t = {frame_idx * dt_frame:.6f} s")
        else:
            time_text.set_text("")
        return [img, time_text]

    ani = animation.FuncAnimation(fig, _update, frames=len(frames), interval=1000.0 / fps, blit=True)
    return fig, ani


# ─── Fast parallel GIF writer ──────────────────────────────────────────────

def _render_one_frame(args):
    """Render a single frame to an RGBA numpy array (for use in process pool)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    (frame_data, frame_idx, cfg_dict, vmax, dt_frame,
     show_time, show_grid, dpi) = args

    extent = [0.0, cfg_dict["nx"] * cfg_dict["dx"],
              cfg_dict["ny"] * cfg_dict["dy"], 0.0]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    img = ax.imshow(frame_data, cmap="coolwarm", vmin=-vmax, vmax=vmax,
                    aspect="auto", extent=extent, interpolation="nearest")
    ax.set_title("Extra Pressure Propagation")
    ax.set_xlabel("x distance (m)")
    ax.set_ylabel("depth y (m)")
    ax.scatter([(cfg_dict["source_ix"] + 0.5) * cfg_dict["dx"]],
               [(cfg_dict["source_iy"] + 0.5) * cfg_dict["dy"]], c="black", s=14)
    fig.colorbar(img, ax=ax, label="Pa")

    if show_grid:
        fig_w_px = fig.get_figwidth() * dpi * 0.75
        fig_h_px = fig.get_figheight() * dpi * 0.75
        nx, ny = cfg_dict["nx"], cfg_dict["ny"]
        sx = _pick_grid_stride(nx, fig_w_px)
        sy = _pick_grid_stride(ny, fig_h_px)
        x_edges = np.arange(0, nx + 1, sx) * cfg_dict["dx"]
        y_edges = np.arange(0, ny + 1, sy) * cfg_dict["dy"]
        ax.set_xticks(x_edges, minor=True)
        ax.set_yticks(y_edges, minor=True)
        ax.grid(which="minor", color="k", alpha=0.25, linewidth=0.3)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color("k")

    if show_time:
        ax.text(0.02, 0.96, f"t = {frame_idx * dt_frame:.6f} s",
                transform=ax.transAxes, color="black", fontsize=10,
                bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"})

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    rgba = buf.copy()
    plt.close(fig)
    return rgba


def save_gif_parallel(result, gif_path: str, fps: int = 25, dpi: int = 140,
                      show_time: bool = True, show_grid: bool = True,
                      max_workers: int = 0):
    """Render GIF frames in parallel across CPU cores, then stitch with Pillow."""
    from concurrent.futures import ProcessPoolExecutor
    from PIL import Image

    cfg: PressureCASimConfig = result["cfg"]
    frames = result["frames_extra"]
    if not frames:
        return

    vmax = float(np.max(np.abs(np.array(frames))))
    vmax = max(vmax, 1e-12)
    dt_frame = cfg.dt * max(1, int(cfg.frame_stride))

    cfg_dict = {
        "nx": cfg.nx, "ny": cfg.ny, "dx": cfg.dx, "dy": cfg.dy,
        "source_ix": cfg.source_ix, "source_iy": cfg.source_iy,
    }

    n_workers = max_workers if max_workers > 0 else max(1, (os.cpu_count() or 4) - 2)
    n_frames = len(frames)

    args_list = [
        (frames[i], i, cfg_dict, vmax, dt_frame, show_time, show_grid, dpi)
        for i in range(n_frames)
    ]

    print(f"  Rendering {n_frames} frames across {n_workers} workers...")
    rendered = [None] * n_frames
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for idx, rgba in enumerate(pool.map(_render_one_frame, args_list)):
            rendered[idx] = Image.fromarray(rgba[:, :, :3])

    duration_ms = int(1000.0 / fps)
    rendered[0].save(
        gif_path,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )
    print(f"  GIF saved: {gif_path}")

