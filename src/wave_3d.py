"""3D acoustic wave propagation engine.

Solves the second-order wave equation on a uniform 3D grid with
depth-varying sound speed, configurable 6-face boundary reflections,
absorption, and a sinusoidal point source.

Array convention: **(nz, ny, nx)** — z is outermost (front→back),
y is height (top→bottom), x is width (left→right).

Three compute backends (selected automatically):

1. **GPU** — fused CUDA kernels via CuPy (fastest, needs NVIDIA GPU)
2. **CPU-Numba** — JIT-compiled parallel loops (fast, needs Numba)
3. **CPU-NumPy** — pure vectorised NumPy (always available)

Two entry points:

- ``WaveStepper3D`` — stateful, step-at-a-time interface for the GUI.
- ``run_wave_3d()`` — batch run returning frame history and traces.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None  # type: ignore[assignment]
    CUPY_AVAILABLE = False

try:
    import numba as nb
    _ncpu = os.cpu_count() or 4
    _nb_threads = max(1, _ncpu - 2)
    nb.config.NUMBA_NUM_THREADS = min(_nb_threads, nb.config.NUMBA_NUM_THREADS)
    NUMBA_AVAILABLE = True
except (ImportError, AttributeError):
    nb = None  # type: ignore[assignment]
    NUMBA_AVAILABLE = False


# ─── Configuration ──────────────────────────────────────────────────────────


@dataclass
class WaveConfig3D:
    """Every parameter the 3D simulator needs."""

    nx: int = 60
    ny: int = 30
    nz: int = 60

    dx: float = 0.05
    dy: float = 0.05
    dz: float = 0.05

    dt: float = 0.0
    steps: int = 500

    ssp_depths_m: Tuple[float, ...] = (0.0, 5.0)
    ssp_speeds_mps: Tuple[float, ...] = (1500.0, 1500.0)

    wave_absorption_per_s: float = 0.05

    boundary_reflect_top: float = -0.98
    boundary_reflect_bottom: float = 0.99
    boundary_reflect_left: float = 0.99
    boundary_reflect_right: float = 0.99
    boundary_reflect_front: float = 0.0
    boundary_reflect_back: float = 0.0

    stencil_order: int = 2

    backend: str = "auto"
    use_float32: bool = True

    source_ix: int = 0
    source_iy: int = 0
    source_iz: int = 0
    source_amplitude_pa: float = 500.0
    source_frequency_hz: float = 1000.0
    source_phase_rad: float = 0.0

    frame_stride: int = 5


# ─── Grid and CFL helpers ──────────────────────────────────────────────────


def auto_resolve_grid_3d(
    frequency_hz: float,
    c_min: float = 1490.0,
    c_max: float = 1540.0,
    width_m: float | None = None,
    height_m: float | None = None,
    depth_m: float | None = None,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    dt: float = 0.0,
    nx: int = 0,
    ny: int = 0,
    nz: int = 0,
    cells_per_wavelength: int = 0,
    cfl_factor: float = 0.40,
    stencil_order: int = 2,
) -> dict:
    """Compute 3D grid / time parameters that avoid aliasing."""
    if cells_per_wavelength <= 0:
        cells_per_wavelength = 5 if stencil_order >= 4 else 10

    wl_min = c_min / max(frequency_hz, 1e-6)

    if dx <= 0:
        dx = wl_min / cells_per_wavelength
    if dy <= 0:
        dy = dx
    if dz <= 0:
        dz = dx

    if dt <= 0:
        inv_sq = 1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2
        dt = cfl_factor / (c_max * inv_sq**0.5)

    def _resolve_axis(size_m, n, d, default_n):
        if size_m is not None and size_m > 0:
            if n <= 0:
                n = max(4, int(np.ceil(size_m / d)))
            return n, n * d
        if n > 0:
            return n, n * d
        return default_n, default_n * d

    nx, width_m = _resolve_axis(width_m, nx, dx, 60)
    ny, height_m = _resolve_axis(height_m, ny, dy, 30)
    nz, depth_m = _resolve_axis(depth_m, nz, dz, 60)

    return dict(dx=dx, dy=dy, dz=dz, dt=dt, nx=nx, ny=ny, nz=nz,
                width_m=width_m, height_m=height_m, depth_m=depth_m)


def check_cfl_3d(cfg: WaveConfig3D, sound_speed: np.ndarray):
    """Validate CFL condition and Nyquist checks for 3D."""
    cmax = float(np.max(sound_speed))
    cmin = float(np.min(sound_speed))
    inv_sq = 1.0 / cfg.dx**2 + 1.0 / cfg.dy**2 + 1.0 / cfg.dz**2
    cfl = cmax * cfg.dt * inv_sq**0.5

    cfl_limit = 0.866 if cfg.stencil_order >= 4 else 1.0
    if cfl >= cfl_limit:
        raise ValueError(
            f"3D CFL unstable: {cfl:.3f} >= {cfl_limit:.3f} "
            f"(stencil_order={cfg.stencil_order}). Reduce dt or increase cell size."
        )

    f = cfg.source_frequency_hz
    if f > 0:
        import warnings
        f_nyquist = 0.5 / cfg.dt
        if f > f_nyquist:
            warnings.warn(
                f"Source {f:.0f} Hz > temporal Nyquist {f_nyquist:.0f} Hz. "
                f"Signal is aliased — reduce dt.", stacklevel=3)
        dx_max = max(cfg.dx, cfg.dy, cfg.dz)
        wl = cmin / f
        cpw = wl / dx_max
        warn_cpw = 3.0 if cfg.stencil_order >= 4 else 5.0
        if cpw < 2.0:
            warnings.warn(
                f"Only {cpw:.1f} cells/wavelength for {f:.0f} Hz — "
                f"grid cannot represent this frequency.", stacklevel=3)
        elif cpw < warn_cpw:
            warnings.warn(
                f"Only {cpw:.1f} cells/λ for {f:.0f} Hz (need ≥{warn_cpw:.0f}). "
                f"Results may be inaccurate.", stacklevel=3)


# ─── Sound speed and source ────────────────────────────────────────────────


def build_sound_speed_grid_3d(cfg: WaveConfig3D) -> np.ndarray:
    """Interpolate SSP onto (nz, ny, nx) grid; speed varies only with y."""
    depths = np.asarray(cfg.ssp_depths_m)
    speeds = np.asarray(cfg.ssp_speeds_mps)
    y_centers = (np.arange(cfg.ny) + 0.5) * cfg.dy
    c_y = np.interp(y_centers, depths, speeds)
    grid = np.empty((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)
    grid[:] = c_y[None, :, None]
    return grid


def _source_value(cfg: WaveConfig3D, t: float) -> float:
    return cfg.source_amplitude_pa * np.sin(
        2.0 * np.pi * cfg.source_frequency_hz * t + cfg.source_phase_rad)


def _sim_dtype(cfg: WaveConfig3D):
    return np.float32 if cfg.use_float32 else np.float64


# ─── Boundary reflection profiles ──────────────────────────────────────────


def _build_boundary_profiles_3d(cfg: WaveConfig3D) -> dict:
    return {
        "top": float(cfg.boundary_reflect_top),
        "bottom": float(cfg.boundary_reflect_bottom),
        "left": float(cfg.boundary_reflect_left),
        "right": float(cfg.boundary_reflect_right),
        "front": float(cfg.boundary_reflect_front),
        "back": float(cfg.boundary_reflect_back),
    }


def _apply_boundaries_3d(p_next, bnd: dict):
    """Apply 6-face reflection boundaries in-place."""
    r_top = bnd["top"]
    r_bot = bnd["bottom"]
    r_left = bnd["left"]
    r_right = bnd["right"]
    r_front = bnd["front"]
    r_back = bnd["back"]

    p_next[:, 0, :] = r_top * p_next[:, 1, :]
    p_next[:, -1, :] = r_bot * p_next[:, -2, :]
    p_next[:, :, 0] = r_left * p_next[:, :, 1]
    p_next[:, :, -1] = r_right * p_next[:, :, -2]
    p_next[0, :, :] = r_front * p_next[1, :, :]
    p_next[-1, :, :] = r_back * p_next[-2, :, :]


# ─── Frame storage (lightweight 3D version) ────────────────────────────────


MAX_FRAME_DIM_3D = 256


def _downsample_3d(vol: np.ndarray, ds: int) -> np.ndarray:
    """Block-mean downsample a 3D array by integer factor."""
    if ds <= 1:
        return vol
    nz, ny, nx = vol.shape
    tz = nz - nz % ds
    ty = ny - ny % ds
    tx = nx - nx % ds
    return vol[:tz, :ty, :tx].reshape(
        tz // ds, ds, ty // ds, ds, tx // ds, ds).mean(axis=(1, 3, 5))


class _FrameSink3D:
    """Accumulates 3D frames in RAM or streams to disk."""

    def __init__(self, nz: int, ny: int, nx: int, n_frames: int,
                 dtype=np.float32, max_ram_bytes: int = 4 * 1024**3,
                 disk_dir: str | None = None,
                 max_dim: int = MAX_FRAME_DIM_3D):
        self.orig_nz, self.orig_ny, self.orig_nx = nz, ny, nx
        self.dtype = np.dtype(dtype)

        self.ds = 1
        while nz // self.ds > max_dim or ny // self.ds > max_dim or nx // self.ds > max_dim:
            self.ds += 1
        if self.ds > 1:
            self.nz = (nz - nz % self.ds) // self.ds
            self.ny = (ny - ny % self.ds) // self.ds
            self.nx = (nx - nx % self.ds) // self.ds
        else:
            self.nz, self.ny, self.nx = nz, ny, nx

        self.count = 0
        self._frame_nbytes = self.nz * self.ny * self.nx * self.dtype.itemsize
        total_bytes = n_frames * self._frame_nbytes

        if total_bytes <= max_ram_bytes:
            self._mode = "ram"
            self._frames: list[np.ndarray] = []
            self._path = None
            self._fh = None
        else:
            self._mode = "disk"
            self._frames = []
            if disk_dir is None:
                disk_dir = os.path.dirname(os.path.abspath(__file__))
            os.makedirs(disk_dir, exist_ok=True)
            self._path = os.path.join(disk_dir, "_frames3d_cache.bin")
            self._fh = open(self._path, "wb")

    def append(self, frame):
        arr = np.asarray(frame, dtype=self.dtype)
        if self.ds > 1:
            arr = _downsample_3d(arr, self.ds).astype(self.dtype)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        if self._mode == "ram":
            self._frames.append(arr)
        else:
            self._fh.write(arr.tobytes())
        self.count += 1

    @property
    def frames(self):
        if self._mode == "ram":
            return self._frames
        if self._fh and not self._fh.closed:
            self._fh.flush()
        return _DiskFrameReader3D(self._path, self.count,
                                   self.nz, self.ny, self.nx, self.dtype)

    @property
    def disk_path(self) -> str | None:
        return self._path

    def write_meta(self):
        if self._path is None:
            return
        meta_path = self._path.replace(".bin", "_meta.json")
        with open(meta_path, "w") as f:
            json.dump({
                "nz": self.nz, "ny": self.ny, "nx": self.nx,
                "orig_nz": self.orig_nz, "orig_ny": self.orig_ny,
                "orig_nx": self.orig_nx,
                "ds": self.ds, "count": self.count,
                "dtype": str(self.dtype),
            }, f)

    def cleanup(self):
        if self._fh and not self._fh.closed:
            self._fh.close()
        if self._path:
            try:
                os.unlink(self._path)
                meta = self._path.replace(".bin", "_meta.json")
                if os.path.exists(meta):
                    os.unlink(meta)
            except OSError:
                pass


class _DiskFrameReader3D:
    """Lazy reader for 3D frames from a flat binary file."""

    def __init__(self, path: str, count: int, nz: int, ny: int, nx: int, dtype):
        self._path = path
        self._count = count
        self._shape = (nz, ny, nx)
        self._dtype = np.dtype(dtype)
        self._frame_nbytes = nz * ny * nx * self._dtype.itemsize
        self._fh = open(path, "rb")

    def __del__(self):
        if hasattr(self, "_fh") and self._fh and not self._fh.closed:
            self._fh.close()

    def __len__(self):
        return self._count

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(self._count))]
        if idx < 0:
            idx += self._count
        if not 0 <= idx < self._count:
            raise IndexError(idx)
        self._fh.seek(idx * self._frame_nbytes)
        buf = self._fh.read(self._frame_nbytes)
        return np.frombuffer(buf, dtype=self._dtype).reshape(self._shape)

    def __iter__(self):
        self._fh.seek(0)
        for _ in range(self._count):
            buf = self._fh.read(self._frame_nbytes)
            yield np.frombuffer(buf, dtype=self._dtype).reshape(self._shape)


# ─── NumPy backend ──────────────────────────────────────────────────────────


def _run_wave_numpy_3d(cfg, sound_speed, frame_sink=None, progress_fn=None):
    dt = _sim_dtype(cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx

    p_prev = np.zeros((nz, ny, nx), dtype=dt)
    p = np.zeros((nz, ny, nx), dtype=dt)
    source_trace = np.zeros(cfg.steps, dtype=dt)
    use_sink = frame_sink is not None
    frames: list = []
    progress_interval = max(1, cfg.steps // 200)

    inv_dx2 = dt(1.0 / cfg.dx**2)
    inv_dy2 = dt(1.0 / cfg.dy**2)
    inv_dz2 = dt(1.0 / cfg.dz**2)
    c2dt2 = (sound_speed.astype(dt, copy=False) * dt(cfg.dt)) ** 2
    sigma_dt = dt(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    bnd = _build_boundary_profiles_3d(cfg)
    use_4th = cfg.stencil_order >= 4 and nz > 4 and ny > 4 and nx > 4
    inv12 = dt(1.0 / 12.0)

    two_m_s = dt(2.0) - sigma_dt
    one_m_s = dt(1.0) - sigma_dt

    for step in range(cfg.steps):
        t = step * cfg.dt
        p_next = np.zeros_like(p)

        # 2nd-order 7-point Laplacian for interior
        lap = (
            (p[1:-1, 1:-1, 2:] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[1:-1, 1:-1, :-2]) * inv_dx2
            + (p[1:-1, 2:, 1:-1] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1]) * inv_dy2
            + (p[2:, 1:-1, 1:-1] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * inv_dz2
        )
        p_next[1:-1, 1:-1, 1:-1] = (
            two_m_s * p[1:-1, 1:-1, 1:-1]
            - one_m_s * p_prev[1:-1, 1:-1, 1:-1]
            + c2dt2[1:-1, 1:-1, 1:-1] * lap
        )

        if use_4th:
            s = p  # alias for brevity
            lap4 = (
                (-s[2:-2, 2:-2, 4:] + dt(16) * s[2:-2, 2:-2, 3:-1] - dt(30) * s[2:-2, 2:-2, 2:-2]
                 + dt(16) * s[2:-2, 2:-2, 1:-3] - s[2:-2, 2:-2, :-4]) * inv_dx2 * inv12
                + (-s[2:-2, 4:, 2:-2] + dt(16) * s[2:-2, 3:-1, 2:-2] - dt(30) * s[2:-2, 2:-2, 2:-2]
                   + dt(16) * s[2:-2, 1:-3, 2:-2] - s[2:-2, :-4, 2:-2]) * inv_dy2 * inv12
                + (-s[4:, 2:-2, 2:-2] + dt(16) * s[3:-1, 2:-2, 2:-2] - dt(30) * s[2:-2, 2:-2, 2:-2]
                   + dt(16) * s[1:-3, 2:-2, 2:-2] - s[:-4, 2:-2, 2:-2]) * inv_dz2 * inv12
            )
            p_next[2:-2, 2:-2, 2:-2] = (
                two_m_s * p[2:-2, 2:-2, 2:-2]
                - one_m_s * p_prev[2:-2, 2:-2, 2:-2]
                + c2dt2[2:-2, 2:-2, 2:-2] * lap4
            )

        _apply_boundaries_3d(p_next, bnd)

        src_val = _source_value(cfg, t)
        p_next[cfg.source_iz, cfg.source_iy, cfg.source_ix] += dt(src_val)
        source_trace[step] = src_val

        p_prev, p = p, p_next
        if step % max(1, cfg.frame_stride) == 0:
            if use_sink:
                frame_sink.append(p.copy())
            else:
                frames.append(p.copy())

        if progress_fn and step % progress_interval == 0:
            progress_fn(step, cfg.steps)

    if progress_fn:
        progress_fn(cfg.steps, cfg.steps)

    return {
        "backend_used": "cpu-numpy-wave-3d",
        "final_pressure": p,
        "frames": frame_sink.frames if use_sink else frames,
        "source_trace": source_trace,
    }


# ─── Numba parallel backend ────────────────────────────────────────────────

_numba_3d_step_2nd = None
_numba_3d_step_4th = None


def _build_numba_3d_kernel_2nd():
    if not NUMBA_AVAILABLE:
        return None

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _step(p, p_prev, p_next, c2dt2,
              inv_dx2, inv_dy2, inv_dz2, two_m_s, one_m_s,
              r_top, r_bot, r_left, r_right, r_front, r_back):
        nz, ny, nx = p.shape
        for k in nb.prange(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    c = p[k, j, i]
                    lap = ((p[k, j, i + 1] - 2.0 * c + p[k, j, i - 1]) * inv_dx2
                           + (p[k, j + 1, i] - 2.0 * c + p[k, j - 1, i]) * inv_dy2
                           + (p[k + 1, j, i] - 2.0 * c + p[k - 1, j, i]) * inv_dz2)
                    p_next[k, j, i] = two_m_s * c - one_m_s * p_prev[k, j, i] + c2dt2[k, j, i] * lap
        # Boundaries
        for k in range(nz):
            for i in range(nx):
                p_next[k, 0, i] = r_top * p_next[k, 1, i]
                p_next[k, ny - 1, i] = r_bot * p_next[k, ny - 2, i]
            for j in range(ny):
                p_next[k, j, 0] = r_left * p_next[k, j, 1]
                p_next[k, j, nx - 1] = r_right * p_next[k, j, nx - 2]
        for j in range(ny):
            for i in range(nx):
                p_next[0, j, i] = r_front * p_next[1, j, i]
                p_next[nz - 1, j, i] = r_back * p_next[nz - 2, j, i]
    return _step


def _build_numba_3d_kernel_4th():
    if not NUMBA_AVAILABLE:
        return None

    @nb.njit(cache=True, fastmath=True, parallel=True)
    def _step(p, p_prev, p_next, c2dt2,
              inv_dx2, inv_dy2, inv_dz2, two_m_s, one_m_s,
              r_top, r_bot, r_left, r_right, r_front, r_back):
        nz, ny, nx = p.shape
        inv12 = 1.0 / 12.0
        for k in nb.prange(1, nz - 1):
            for j in range(1, ny - 1):
                for i in range(1, nx - 1):
                    c = p[k, j, i]
                    if (k >= 2 and k < nz - 2 and
                            j >= 2 and j < ny - 2 and
                            i >= 2 and i < nx - 2):
                        lap = ((-p[k, j, i + 2] + 16.0 * p[k, j, i + 1] - 30.0 * c
                                + 16.0 * p[k, j, i - 1] - p[k, j, i - 2]) * inv_dx2 * inv12
                               + (-p[k, j + 2, i] + 16.0 * p[k, j + 1, i] - 30.0 * c
                                  + 16.0 * p[k, j - 1, i] - p[k, j - 2, i]) * inv_dy2 * inv12
                               + (-p[k + 2, j, i] + 16.0 * p[k + 1, j, i] - 30.0 * c
                                  + 16.0 * p[k - 1, j, i] - p[k - 2, j, i]) * inv_dz2 * inv12)
                    else:
                        lap = ((p[k, j, i + 1] - 2.0 * c + p[k, j, i - 1]) * inv_dx2
                               + (p[k, j + 1, i] - 2.0 * c + p[k, j - 1, i]) * inv_dy2
                               + (p[k + 1, j, i] - 2.0 * c + p[k - 1, j, i]) * inv_dz2)
                    p_next[k, j, i] = two_m_s * c - one_m_s * p_prev[k, j, i] + c2dt2[k, j, i] * lap
        for k in range(nz):
            for i in range(nx):
                p_next[k, 0, i] = r_top * p_next[k, 1, i]
                p_next[k, ny - 1, i] = r_bot * p_next[k, ny - 2, i]
            for j in range(ny):
                p_next[k, j, 0] = r_left * p_next[k, j, 1]
                p_next[k, j, nx - 1] = r_right * p_next[k, j, nx - 2]
        for j in range(ny):
            for i in range(nx):
                p_next[0, j, i] = r_front * p_next[1, j, i]
                p_next[nz - 1, j, i] = r_back * p_next[nz - 2, j, i]
    return _step


def _get_numba_3d_step(stencil_order: int = 2):
    global _numba_3d_step_2nd, _numba_3d_step_4th
    if stencil_order >= 4:
        if _numba_3d_step_4th is None:
            _numba_3d_step_4th = _build_numba_3d_kernel_4th()
        return _numba_3d_step_4th
    if _numba_3d_step_2nd is None:
        _numba_3d_step_2nd = _build_numba_3d_kernel_2nd()
    return _numba_3d_step_2nd


def _run_wave_numba_3d(cfg, sound_speed, frame_sink=None, progress_fn=None):
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba not available.")
    kernel = _get_numba_3d_step(cfg.stencil_order)
    if kernel is None:
        raise RuntimeError("Failed to compile 3D Numba kernel.")

    dt = _sim_dtype(cfg)
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx
    p_prev = np.zeros((nz, ny, nx), dtype=dt)
    p = np.zeros((nz, ny, nx), dtype=dt)
    p_next = np.zeros((nz, ny, nx), dtype=dt)
    use_sink = frame_sink is not None
    frames: list = []
    source_trace = np.zeros(cfg.steps, dtype=dt)
    progress_interval = max(1, cfg.steps // 200)

    inv_dx2 = dt(1.0 / cfg.dx**2)
    inv_dy2 = dt(1.0 / cfg.dy**2)
    inv_dz2 = dt(1.0 / cfg.dz**2)
    c2dt2 = (sound_speed.astype(dt, copy=False) * dt(cfg.dt)) ** 2
    sigma_dt = float(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    two_m_s = 2.0 - sigma_dt
    one_m_s = 1.0 - sigma_dt
    bnd = _build_boundary_profiles_3d(cfg)

    # Warm up JIT
    kernel(p, p_prev, p_next, c2dt2,
           inv_dx2, inv_dy2, inv_dz2, two_m_s, one_m_s,
           bnd["top"], bnd["bottom"], bnd["left"], bnd["right"],
           bnd["front"], bnd["back"])
    p.fill(0); p_prev.fill(0); p_next.fill(0)

    for step in range(cfg.steps):
        t = step * cfg.dt
        kernel(p, p_prev, p_next, c2dt2,
               inv_dx2, inv_dy2, inv_dz2, two_m_s, one_m_s,
               bnd["top"], bnd["bottom"], bnd["left"], bnd["right"],
               bnd["front"], bnd["back"])

        src_val = dt(_source_value(cfg, t))
        p_next[cfg.source_iz, cfg.source_iy, cfg.source_ix] += src_val
        source_trace[step] = src_val

        p_prev, p, p_next = p, p_next, p_prev
        if step % max(1, cfg.frame_stride) == 0:
            if use_sink:
                frame_sink.append(p.copy())
            else:
                frames.append(p.copy())

        if progress_fn and step % progress_interval == 0:
            progress_fn(step, cfg.steps)

    if progress_fn:
        progress_fn(cfg.steps, cfg.steps)

    return {
        "backend_used": f"cpu-numba-wave-3d-{nb.config.NUMBA_NUM_THREADS}threads",
        "final_pressure": p,
        "frames": frame_sink.frames if use_sink else frames,
        "source_trace": source_trace,
    }


# ─── CuPy fused CUDA backend ───────────────────────────────────────────────

_WAVE_3D_CUDA_2ND = r'''
extern "C" __global__
void wave3d_step_2nd_{T}(
    const {F}* __restrict__ p,
    const {F}* __restrict__ p_prev,
    const {F}* __restrict__ c2dt2,
    {F}* __restrict__ p_next,
    const int nz, const int ny, const int nx,
    const {F} inv_dx2, const {F} inv_dy2, const {F} inv_dz2,
    const {F} two_m_s, const {F} one_m_s
) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nz * ny * nx;
    if (idx >= total) return;
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);
    if (k < 1 || k >= nz - 1 || j < 1 || j >= ny - 1 || i < 1 || i >= nx - 1) return;

    int nxy = ny * nx;
    {F} c = p[idx];
    {F} lap = (p[idx + 1] - ({F})2.0 * c + p[idx - 1]) * inv_dx2
            + (p[idx + nx] - ({F})2.0 * c + p[idx - nx]) * inv_dy2
            + (p[idx + nxy] - ({F})2.0 * c + p[idx - nxy]) * inv_dz2;
    p_next[idx] = two_m_s * c - one_m_s * p_prev[idx] + c2dt2[idx] * lap;
}}
'''

_WAVE_3D_CUDA_4TH = r'''
extern "C" __global__
void wave3d_step_4th_{T}(
    const {F}* __restrict__ p,
    const {F}* __restrict__ p_prev,
    const {F}* __restrict__ c2dt2,
    {F}* __restrict__ p_next,
    const int nz, const int ny, const int nx,
    const {F} inv_dx2, const {F} inv_dy2, const {F} inv_dz2,
    const {F} two_m_s, const {F} one_m_s
) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = nz * ny * nx;
    if (idx >= total) return;
    int i = idx % nx;
    int j = (idx / nx) % ny;
    int k = idx / (nx * ny);
    if (k < 1 || k >= nz - 1 || j < 1 || j >= ny - 1 || i < 1 || i >= nx - 1) return;

    int nxy = ny * nx;
    {F} c = p[idx];
    {F} lap;
    if (k >= 2 && k < nz - 2 && j >= 2 && j < ny - 2 && i >= 2 && i < nx - 2) {{
        {F} inv12 = ({F})1.0 / ({F})12.0;
        lap = (-p[idx + 2] + ({F})16.0 * p[idx + 1] - ({F})30.0 * c
               + ({F})16.0 * p[idx - 1] - p[idx - 2]) * inv_dx2 * inv12
            + (-p[idx + 2 * nx] + ({F})16.0 * p[idx + nx] - ({F})30.0 * c
               + ({F})16.0 * p[idx - nx] - p[idx - 2 * nx]) * inv_dy2 * inv12
            + (-p[idx + 2 * nxy] + ({F})16.0 * p[idx + nxy] - ({F})30.0 * c
               + ({F})16.0 * p[idx - nxy] - p[idx - 2 * nxy]) * inv_dz2 * inv12;
    }} else {{
        lap = (p[idx + 1] - ({F})2.0 * c + p[idx - 1]) * inv_dx2
            + (p[idx + nx] - ({F})2.0 * c + p[idx - nx]) * inv_dy2
            + (p[idx + nxy] - ({F})2.0 * c + p[idx - nxy]) * inv_dz2;
    }}
    p_next[idx] = two_m_s * c - one_m_s * p_prev[idx] + c2dt2[idx] * lap;
}}
'''

_WAVE_3D_BOUNDARY_CUDA = r'''
extern "C" __global__
void wave3d_boundary_{T}(
    {F}* __restrict__ pn,
    const int nz, const int ny, const int nx,
    const {F} r_top, const {F} r_bot,
    const {F} r_left, const {F} r_right,
    const {F} r_front, const {F} r_back,
    const int src_iz, const int src_iy, const int src_ix,
    const {F} src_val
) {{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int nxy = ny * nx;
    int total_bnd = 2 * nz * nx + 2 * nz * ny + 2 * ny * nx;
    if (idx >= total_bnd) return;

    int off = 0;
    // Top face (y=0):  nz * nx entries
    if (idx < off + nz * nx) {{
        int local = idx - off;
        int k = local / nx;
        int i = local % nx;
        pn[k * nxy + 0 * nx + i] = r_top * pn[k * nxy + 1 * nx + i];
        goto src;
    }}
    off += nz * nx;
    // Bottom face (y=ny-1): nz * nx entries
    if (idx < off + nz * nx) {{
        int local = idx - off;
        int k = local / nx;
        int i = local % nx;
        pn[k * nxy + (ny - 1) * nx + i] = r_bot * pn[k * nxy + (ny - 2) * nx + i];
        goto src;
    }}
    off += nz * nx;
    // Left face (x=0): nz * ny entries
    if (idx < off + nz * ny) {{
        int local = idx - off;
        int k = local / ny;
        int j = local % ny;
        pn[k * nxy + j * nx + 0] = r_left * pn[k * nxy + j * nx + 1];
        goto src;
    }}
    off += nz * ny;
    // Right face (x=nx-1): nz * ny entries
    if (idx < off + nz * ny) {{
        int local = idx - off;
        int k = local / ny;
        int j = local % ny;
        pn[k * nxy + j * nx + nx - 1] = r_right * pn[k * nxy + j * nx + nx - 2];
        goto src;
    }}
    off += nz * ny;
    // Front face (z=0): ny * nx entries
    if (idx < off + ny * nx) {{
        int local = idx - off;
        int j = local / nx;
        int i = local % nx;
        pn[0 * nxy + j * nx + i] = r_front * pn[1 * nxy + j * nx + i];
        goto src;
    }}
    off += ny * nx;
    // Back face (z=nz-1): ny * nx entries
    if (idx < off + ny * nx) {{
        int local = idx - off;
        int j = local / nx;
        int i = local % nx;
        pn[(nz - 1) * nxy + j * nx + i] = r_back * pn[(nz - 2) * nxy + j * nx + i];
    }}

src:
    if (idx == 0) {{
        pn[src_iz * nxy + src_iy * nx + src_ix] += src_val;
    }}
}}
'''

_cuda_3d_kernels: dict = {}


def _get_cuda_3d_wave_kernel(use_float32: bool, stencil_order: int = 2):
    is_4th = stencil_order >= 4
    T = "f32" if use_float32 else "f64"
    F = "float" if use_float32 else "double"
    key = f"{'4th_' if is_4th else ''}{T}"
    if key not in _cuda_3d_kernels:
        tmpl = _WAVE_3D_CUDA_4TH if is_4th else _WAVE_3D_CUDA_2ND
        name = f"wave3d_step_{'4th_' if is_4th else ''}{T}"
        src = tmpl.format(T=T, F=F)
        _cuda_3d_kernels[key] = cp.RawKernel(src, name)
    return _cuda_3d_kernels[key]


def _get_cuda_3d_boundary_kernel(use_float32: bool):
    T = "f32" if use_float32 else "f64"
    F = "float" if use_float32 else "double"
    key = f"bnd3d_{T}"
    if key not in _cuda_3d_kernels:
        src = _WAVE_3D_BOUNDARY_CUDA.format(T=T, F=F)
        _cuda_3d_kernels[key] = cp.RawKernel(src, f"wave3d_boundary_{T}")
    return _cuda_3d_kernels[key]


def _run_wave_cupy_3d(cfg, sound_speed, frame_sink=None, progress_fn=None):
    if not CUPY_AVAILABLE:
        raise RuntimeError("CuPy not available.")
    _ = cp.cuda.runtime.getDeviceCount()

    dt_np = _sim_dtype(cfg)
    dt_cp = cp.float32 if dt_np == np.float32 else cp.float64
    nz, ny, nx = cfg.nz, cfg.ny, cfg.nx

    kernel = _get_cuda_3d_wave_kernel(cfg.use_float32, cfg.stencil_order)
    bnd_kernel = _get_cuda_3d_boundary_kernel(cfg.use_float32)

    total_cells = nz * ny * nx
    block = 256
    grid_k = (total_cells + block - 1) // block

    total_bnd = 2 * nz * nx + 2 * nz * ny + 2 * ny * nx
    bnd_block = min(256, total_bnd)
    bnd_grid = (total_bnd + bnd_block - 1) // bnd_block

    p_prev = cp.zeros((nz, ny, nx), dtype=dt_cp)
    p = cp.zeros((nz, ny, nx), dtype=dt_cp)
    p_next = cp.zeros((nz, ny, nx), dtype=dt_cp)
    c2dt2 = (cp.asarray(sound_speed, dtype=dt_cp) * dt_cp(cfg.dt)) ** 2

    inv_dx2 = dt_np(1.0 / cfg.dx**2)
    inv_dy2 = dt_np(1.0 / cfg.dy**2)
    inv_dz2 = dt_np(1.0 / cfg.dz**2)
    sigma_dt = float(max(0.0, cfg.wave_absorption_per_s * cfg.dt))
    two_m_s = dt_np(2.0 - sigma_dt)
    one_m_s = dt_np(1.0 - sigma_dt)

    bnd = _build_boundary_profiles_3d(cfg)
    nz_i, ny_i, nx_i = np.int32(nz), np.int32(ny), np.int32(nx)

    use_sink = frame_sink is not None
    frames: list = []
    source_trace = np.zeros(cfg.steps, dtype=dt_np)
    progress_interval = max(1, cfg.steps // 200)

    for step in range(cfg.steps):
        t = step * cfg.dt

        kernel((grid_k,), (block,),
               (p, p_prev, c2dt2, p_next,
                nz_i, ny_i, nx_i,
                inv_dx2, inv_dy2, inv_dz2,
                two_m_s, one_m_s))

        src_val = _source_value(cfg, t)
        bnd_kernel((bnd_grid,), (bnd_block,),
                   (p_next, nz_i, ny_i, nx_i,
                    dt_np(bnd["top"]), dt_np(bnd["bottom"]),
                    dt_np(bnd["left"]), dt_np(bnd["right"]),
                    dt_np(bnd["front"]), dt_np(bnd["back"]),
                    np.int32(cfg.source_iz), np.int32(cfg.source_iy),
                    np.int32(cfg.source_ix), dt_np(src_val)))

        source_trace[step] = src_val
        p_prev, p, p_next = p, p_next, p_prev

        if step % max(1, cfg.frame_stride) == 0:
            frame_np = cp.asnumpy(p)
            if use_sink:
                frame_sink.append(frame_np)
            else:
                frames.append(frame_np)

        if progress_fn and step % progress_interval == 0:
            progress_fn(step, cfg.steps)

    if progress_fn:
        progress_fn(cfg.steps, cfg.steps)

    cp.cuda.Stream.null.synchronize()
    dev = cp.cuda.Device(0)
    dev_name = dev.attributes.get("DeviceName", f"GPU-{dev.id}")
    return {
        "backend_used": f"gpu-cupy-fused-wave-3d ({dev_name})",
        "final_pressure": cp.asnumpy(p),
        "frames": frame_sink.frames if use_sink else frames,
        "source_trace": source_trace,
    }


# ─── WaveStepper3D (interactive single-step interface) ─────────────────────


class WaveStepper3D:
    """Wraps 3D wave state for interactive stepping (GUI use)."""

    def __init__(self, cfg: WaveConfig3D):
        if cfg.nx <= 2 or cfg.ny <= 2 or cfg.nz <= 2:
            raise ValueError("Grid must be at least 3x3x3.")
        if cfg.dt <= 0:
            raise ValueError("dt must be > 0.")
        if not (0 <= cfg.source_ix < cfg.nx and
                0 <= cfg.source_iy < cfg.ny and
                0 <= cfg.source_iz < cfg.nz):
            raise ValueError("Source index outside grid.")

        self.cfg = cfg
        self._step_count = 0
        self._dtype = _sim_dtype(cfg)
        self._sound_speed = build_sound_speed_grid_3d(cfg)
        check_cfl_3d(cfg, self._sound_speed)

        bnd = _build_boundary_profiles_3d(cfg)
        self._bnd = bnd
        self._backend_name = "unknown"
        self._use_gpu = False

        requested = str(cfg.backend).strip().lower()

        if requested in ("auto", "gpu") and CUPY_AVAILABLE:
            try:
                self._init_gpu(bnd)
                return
            except Exception:
                if requested == "gpu":
                    raise

        if requested in ("auto", "cpu") and NUMBA_AVAILABLE:
            try:
                self._init_numba(bnd)
                return
            except Exception:
                pass

        self._init_numpy(bnd)

    # ── NumPy init ──

    def _init_numpy(self, bnd):
        dt = self._dtype
        nz, ny, nx = self.cfg.nz, self.cfg.ny, self.cfg.nx
        self._p_prev = np.zeros((nz, ny, nx), dtype=dt)
        self._p = np.zeros((nz, ny, nx), dtype=dt)
        self._inv_dx2 = dt(1.0 / self.cfg.dx**2)
        self._inv_dy2 = dt(1.0 / self.cfg.dy**2)
        self._inv_dz2 = dt(1.0 / self.cfg.dz**2)
        self._c2dt2 = (self._sound_speed.astype(dt, copy=False) * dt(self.cfg.dt)) ** 2
        sigma_dt = dt(max(0.0, self.cfg.wave_absorption_per_s * self.cfg.dt))
        self._two_m_s = dt(2.0) - sigma_dt
        self._one_m_s = dt(1.0) - sigma_dt
        self._backend_name = "cpu-numpy-wave-3d"
        self._step_fn = self._step_numpy

    # ── Numba init ──

    def _init_numba(self, bnd):
        kernel = _get_numba_3d_step(self.cfg.stencil_order)
        if kernel is None:
            raise RuntimeError("Numba 3D kernel compile failed")
        self._init_numpy(bnd)
        self._p_next_buf = np.zeros_like(self._p)
        self._nb_kernel = kernel
        # Warm up
        self._nb_kernel(
            self._p, self._p_prev, self._p_next_buf,
            self._c2dt2, self._inv_dx2, self._inv_dy2, self._inv_dz2,
            float(self._two_m_s), float(self._one_m_s),
            bnd["top"], bnd["bottom"], bnd["left"], bnd["right"],
            bnd["front"], bnd["back"])
        self._p.fill(0); self._p_prev.fill(0); self._p_next_buf.fill(0)
        self._backend_name = f"cpu-numba-wave-3d-{nb.config.NUMBA_NUM_THREADS}threads"
        self._step_fn = self._step_numba

    # ── GPU init ──

    def _init_gpu(self, bnd):
        if not CUPY_AVAILABLE:
            raise RuntimeError("CuPy not available")
        _ = cp.cuda.runtime.getDeviceCount()
        dt_np = self._dtype
        dt_cp = cp.float32 if dt_np == np.float32 else cp.float64
        self._dt_cp = dt_cp

        nz, ny, nx = self.cfg.nz, self.cfg.ny, self.cfg.nx
        self._gpu_kernel = _get_cuda_3d_wave_kernel(self.cfg.use_float32, self.cfg.stencil_order)
        self._gpu_bnd_kernel = _get_cuda_3d_boundary_kernel(self.cfg.use_float32)

        self._gpu_p_prev = cp.zeros((nz, ny, nx), dtype=dt_cp)
        self._gpu_p = cp.zeros((nz, ny, nx), dtype=dt_cp)
        self._gpu_p_next = cp.zeros((nz, ny, nx), dtype=dt_cp)
        self._gpu_c2dt2 = (cp.asarray(self._sound_speed, dtype=dt_cp) * dt_cp(self.cfg.dt)) ** 2

        sigma_dt = float(max(0.0, self.cfg.wave_absorption_per_s * self.cfg.dt))
        self._gpu_two_m_s = dt_np(2.0 - sigma_dt)
        self._gpu_one_m_s = dt_np(1.0 - sigma_dt)
        self._gpu_inv_dx2 = dt_np(1.0 / self.cfg.dx**2)
        self._gpu_inv_dy2 = dt_np(1.0 / self.cfg.dy**2)
        self._gpu_inv_dz2 = dt_np(1.0 / self.cfg.dz**2)
        self._gpu_nz = np.int32(nz)
        self._gpu_ny = np.int32(ny)
        self._gpu_nx = np.int32(nx)

        total = nz * ny * nx
        self._gpu_block = 256
        self._gpu_grid = (total + self._gpu_block - 1) // self._gpu_block

        total_bnd = 2 * nz * nx + 2 * nz * ny + 2 * ny * nx
        self._gpu_bnd_block = min(256, total_bnd)
        self._gpu_bnd_grid = (total_bnd + self._gpu_bnd_block - 1) // self._gpu_bnd_block

        self._gpu_bnd_args_static = (
            self._gpu_nz, self._gpu_ny, self._gpu_nx,
            dt_np(bnd["top"]), dt_np(bnd["bottom"]),
            dt_np(bnd["left"]), dt_np(bnd["right"]),
            dt_np(bnd["front"]), dt_np(bnd["back"]),
            np.int32(self.cfg.source_iz),
            np.int32(self.cfg.source_iy),
            np.int32(self.cfg.source_ix),
        )

        dev = cp.cuda.Device(0)
        dev_name = dev.attributes.get("DeviceName", f"GPU-{dev.id}")
        self._backend_name = f"gpu-cupy-fused-3d ({dev_name})"
        self._use_gpu = True
        self._step_fn = self._step_gpu

    # ── Step implementations ──

    def _step_numpy(self):
        cfg = self.cfg
        dt = self._dtype
        t = self._step_count * cfg.dt
        p, pp = self._p, self._p_prev

        p_next = np.zeros_like(p)
        lap = (
            (p[1:-1, 1:-1, 2:] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[1:-1, 1:-1, :-2]) * self._inv_dx2
            + (p[1:-1, 2:, 1:-1] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[1:-1, :-2, 1:-1]) * self._inv_dy2
            + (p[2:, 1:-1, 1:-1] - dt(2.0) * p[1:-1, 1:-1, 1:-1] + p[:-2, 1:-1, 1:-1]) * self._inv_dz2
        )
        p_next[1:-1, 1:-1, 1:-1] = (
            self._two_m_s * p[1:-1, 1:-1, 1:-1]
            - self._one_m_s * pp[1:-1, 1:-1, 1:-1]
            + self._c2dt2[1:-1, 1:-1, 1:-1] * lap
        )

        if cfg.stencil_order >= 4 and cfg.nz > 4 and cfg.ny > 4 and cfg.nx > 4:
            inv12 = dt(1.0 / 12.0)
            s = p
            lap4 = (
                (-s[2:-2, 2:-2, 4:] + dt(16) * s[2:-2, 2:-2, 3:-1] - dt(30) * s[2:-2, 2:-2, 2:-2]
                 + dt(16) * s[2:-2, 2:-2, 1:-3] - s[2:-2, 2:-2, :-4]) * self._inv_dx2 * inv12
                + (-s[2:-2, 4:, 2:-2] + dt(16) * s[2:-2, 3:-1, 2:-2] - dt(30) * s[2:-2, 2:-2, 2:-2]
                   + dt(16) * s[2:-2, 1:-3, 2:-2] - s[2:-2, :-4, 2:-2]) * self._inv_dy2 * inv12
                + (-s[4:, 2:-2, 2:-2] + dt(16) * s[3:-1, 2:-2, 2:-2] - dt(30) * s[2:-2, 2:-2, 2:-2]
                   + dt(16) * s[1:-3, 2:-2, 2:-2] - s[:-4, 2:-2, 2:-2]) * self._inv_dz2 * inv12
            )
            p_next[2:-2, 2:-2, 2:-2] = (
                self._two_m_s * p[2:-2, 2:-2, 2:-2]
                - self._one_m_s * pp[2:-2, 2:-2, 2:-2]
                + self._c2dt2[2:-2, 2:-2, 2:-2] * lap4
            )

        _apply_boundaries_3d(p_next, self._bnd)
        src_val = _source_value(cfg, t)
        p_next[cfg.source_iz, cfg.source_iy, cfg.source_ix] += dt(src_val)

        self._p_prev = p
        self._p = p_next
        return src_val

    def _step_numba(self):
        cfg = self.cfg
        t = self._step_count * cfg.dt
        bnd = self._bnd

        self._nb_kernel(
            self._p, self._p_prev, self._p_next_buf,
            self._c2dt2, self._inv_dx2, self._inv_dy2, self._inv_dz2,
            float(self._two_m_s), float(self._one_m_s),
            bnd["top"], bnd["bottom"], bnd["left"], bnd["right"],
            bnd["front"], bnd["back"])

        src_val = self._dtype(_source_value(cfg, t))
        self._p_next_buf[cfg.source_iz, cfg.source_iy, cfg.source_ix] += src_val

        self._p_prev, self._p, self._p_next_buf = self._p, self._p_next_buf, self._p_prev
        return float(src_val)

    def _step_gpu(self):
        cfg = self.cfg
        t = self._step_count * cfg.dt
        src_val = _source_value(cfg, t)

        self._gpu_kernel(
            (self._gpu_grid,), (self._gpu_block,),
            (self._gpu_p, self._gpu_p_prev, self._gpu_c2dt2, self._gpu_p_next,
             self._gpu_nz, self._gpu_ny, self._gpu_nx,
             self._gpu_inv_dx2, self._gpu_inv_dy2, self._gpu_inv_dz2,
             self._gpu_two_m_s, self._gpu_one_m_s))

        self._gpu_bnd_kernel(
            (self._gpu_bnd_grid,), (self._gpu_bnd_block,),
            (self._gpu_p_next, *self._gpu_bnd_args_static,
             self._dt_cp(src_val)))

        self._gpu_p_prev, self._gpu_p, self._gpu_p_next = (
            self._gpu_p, self._gpu_p_next, self._gpu_p_prev)
        return float(src_val)

    # ── Public API ──

    def step(self) -> np.ndarray:
        """Advance one timestep; return pressure (nz, ny, nx)."""
        self._step_fn()
        self._step_count += 1
        return self.pressure

    def step_n(self, n: int,
               probe_cells: list[tuple[int, int, int]] | None = None,
               probe_stride: int = 1,
               ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Advance *n* steps. Probes are (ix, iy, iz) tuples."""
        cfg = self.cfg
        n_probes = len(probe_cells) if probe_cells else 0
        probe_stride = max(1, probe_stride)

        if n_probes:
            n_samples = (n + probe_stride - 1) // probe_stride
            probe_vals = np.empty((n_samples, n_probes), dtype=np.float64)
            probe_ts = np.empty(n_samples, dtype=np.float64)
        else:
            probe_vals = probe_ts = None

        if self._use_gpu:
            self._step_n_gpu(n, probe_cells, probe_vals, probe_ts, probe_stride)
        elif hasattr(self, "_nb_kernel"):
            self._step_n_numba(n, probe_cells, probe_vals, probe_ts, probe_stride)
        else:
            self._step_n_numpy(n, probe_cells, probe_vals, probe_ts, probe_stride)

        return self.pressure, probe_vals, probe_ts

    def _step_n_gpu(self, n, probe_cells, probe_vals, probe_ts, stride):
        cfg = self.cfg
        int_kernel = self._gpu_kernel
        bnd_kernel = self._gpu_bnd_kernel
        igrid, iblock = (self._gpu_grid,), (self._gpu_block,)
        bgrid, bblock = (self._gpu_bnd_grid,), (self._gpu_bnd_block,)
        dt_cast = self._dt_cp
        p, pp, pn = self._gpu_p, self._gpu_p_prev, self._gpu_p_next
        c2dt2 = self._gpu_c2dt2
        gnz, gny, gnx = self._gpu_nz, self._gpu_ny, self._gpu_nx
        bnd_static = self._gpu_bnd_args_static
        has_probes = probe_vals is not None

        step_indices = np.arange(self._step_count, self._step_count + n)
        t_arr = step_indices * cfg.dt
        src_all = cfg.source_amplitude_pa * np.sin(
            2.0 * np.pi * cfg.source_frequency_hz * t_arr + cfg.source_phase_rad)

        if has_probes:
            iz_arr = cp.array([iz for ix, iy, iz in probe_cells], dtype=cp.int64)
            iy_arr = cp.array([iy for ix, iy, iz in probe_cells], dtype=cp.int64)
            ix_arr = cp.array([ix for ix, iy, iz in probe_cells], dtype=cp.int64)
            n_samples = probe_vals.shape[0]
            gpu_buf = cp.empty((n_samples, len(probe_cells)), dtype=cp.float64)

        sample_idx = 0
        for i in range(n):
            src_val = dt_cast(float(src_all[i]))

            int_kernel(igrid, iblock,
                       (p, pp, c2dt2, pn,
                        gnz, gny, gnx,
                        self._gpu_inv_dx2, self._gpu_inv_dy2, self._gpu_inv_dz2,
                        self._gpu_two_m_s, self._gpu_one_m_s))
            bnd_kernel(bgrid, bblock,
                       (pn, *bnd_static, src_val))

            pp, p, pn = p, pn, pp
            self._step_count += 1

            if has_probes and (i % stride == stride - 1 or i == n - 1):
                gpu_buf[sample_idx, :] = p[iz_arr, iy_arr, ix_arr]
                sample_idx += 1

        self._gpu_p_prev, self._gpu_p, self._gpu_p_next = pp, p, pn
        cp.cuda.Stream.null.synchronize()

        if has_probes:
            probe_vals[:sample_idx] = cp.asnumpy(gpu_buf[:sample_idx])
            base = self._step_count - n
            si = 0
            for i in range(n):
                if i % stride == stride - 1 or i == n - 1:
                    probe_ts[si] = (base + i + 1) * cfg.dt
                    si += 1

    def _step_n_numba(self, n, probe_cells, probe_vals, probe_ts, stride):
        cfg = self.cfg
        kernel = self._nb_kernel
        bnd = self._bnd
        has_probes = probe_vals is not None
        sample_idx = 0
        for i in range(n):
            t = self._step_count * cfg.dt
            kernel(self._p, self._p_prev, self._p_next_buf,
                   self._c2dt2, self._inv_dx2, self._inv_dy2, self._inv_dz2,
                   float(self._two_m_s), float(self._one_m_s),
                   bnd["top"], bnd["bottom"], bnd["left"], bnd["right"],
                   bnd["front"], bnd["back"])
            sv = self._dtype(_source_value(cfg, t))
            self._p_next_buf[cfg.source_iz, cfg.source_iy, cfg.source_ix] += sv
            self._p_prev, self._p, self._p_next_buf = (
                self._p, self._p_next_buf, self._p_prev)
            self._step_count += 1
            if has_probes and (i % stride == stride - 1 or i == n - 1):
                probe_ts[sample_idx] = self._step_count * cfg.dt
                for j, (ix, iy, iz) in enumerate(probe_cells):
                    probe_vals[sample_idx, j] = self._p[iz, iy, ix]
                sample_idx += 1

    def _step_n_numpy(self, n, probe_cells, probe_vals, probe_ts, stride):
        has_probes = probe_vals is not None
        sample_idx = 0
        for i in range(n):
            self._step_numpy()
            self._step_count += 1
            if has_probes and (i % stride == stride - 1 or i == n - 1):
                probe_ts[sample_idx] = self._step_count * self.cfg.dt
                for j, (ix, iy, iz) in enumerate(probe_cells):
                    probe_vals[sample_idx, j] = self._p[iz, iy, ix]
                sample_idx += 1

    @property
    def pressure(self) -> np.ndarray:
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


# ─── Batch entry point ─────────────────────────────────────────────────────


def run_wave_3d(cfg: WaveConfig3D, progress_fn=None,
                frame_dir: str | None = None) -> dict:
    """Run the full 3D simulation and return results."""
    if cfg.nx <= 2 or cfg.ny <= 2 or cfg.nz <= 2:
        raise ValueError("Grid must be at least 3x3x3.")
    if cfg.dt <= 0:
        raise ValueError("dt must be > 0.")
    if cfg.steps <= 0:
        raise ValueError("steps must be > 0.")
    if not (0 <= cfg.source_ix < cfg.nx and
            0 <= cfg.source_iy < cfg.ny and
            0 <= cfg.source_iz < cfg.nz):
        raise ValueError("Source index outside grid.")

    sound_speed = build_sound_speed_grid_3d(cfg)
    check_cfl_3d(cfg, sound_speed)

    n_frames_est = cfg.steps // max(1, cfg.frame_stride) + 1
    dtype = np.float32 if cfg.use_float32 else np.float64
    sink = _FrameSink3D(cfg.nz, cfg.ny, cfg.nx, n_frames_est,
                        dtype=dtype, disk_dir=frame_dir)
    if sink.ds > 1:
        print(f"  3D frames downsampled {sink.ds}x: "
              f"{cfg.nx}x{cfg.ny}x{cfg.nz} -> {sink.nx}x{sink.ny}x{sink.nz}")
    est_gb = n_frames_est * sink._frame_nbytes / (1024**3)
    if sink._mode == "disk":
        print(f"  Frames will stream to disk ({est_gb:.1f} GB): {sink.disk_path}")

    requested = str(cfg.backend).strip().lower()
    if requested not in ("auto", "cpu", "gpu"):
        raise ValueError("backend must be one of: auto, cpu, gpu")

    runner = None
    if requested in ("auto", "gpu") and CUPY_AVAILABLE:
        try:
            _ = cp.cuda.runtime.getDeviceCount()
            runner = _run_wave_cupy_3d
        except Exception:
            if requested == "gpu":
                raise

    if runner is None and requested in ("auto", "cpu") and NUMBA_AVAILABLE:
        runner = _run_wave_numba_3d

    if runner is None:
        runner = _run_wave_numpy_3d

    result = runner(cfg, sound_speed, frame_sink=sink, progress_fn=progress_fn)
    sink.write_meta()
    result["_frame_sink"] = sink
    result["sound_speed"] = sound_speed
    return result
