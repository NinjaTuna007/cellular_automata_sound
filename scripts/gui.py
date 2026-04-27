#!/usr/bin/env python3
"""Real-time interactive viewer for the underwater sound simulator.

Opens a Pygame window that shows the pressure field as a live heatmap
(coolwarm: blue = negative, white = zero, red = positive).  Click
anywhere on the grid to drop a virtual hydrophone — its received
signal appears in a combined plot panel on the right, overlaid with
the analytical source waveform for comparison.

Usage
-----
::

    python scripts/gui.py                                        # default config
    python scripts/gui.py --config configs/24khz_all_reflect.yaml
    python scripts/gui.py --config configs/default.yaml --mode playback

Modes
-----
- **live** (default): simulation runs on-the-fly.  Press ``+`` to
  increase steps-per-frame and keep the GPU busy.
- **playback**: pre-computes all frames, then lets you scrub/rewind
  like a video.

Controls
--------
  Space        Play / pause
  Right arrow  Step forward (when paused)
  Left arrow   Step backward (playback mode, when paused)
  R            Reset to step 0
  +/=          Increase simulation speed
  -            Decrease simulation speed
  Click grid   Place a probe (pressure time-series)
  Right-click  Remove probe under cursor
  Click legend Remove that probe's plot
  Esc / Q      Quit

Architecture
------------
- **Main loop**: Pygame event handling → ``stepper.step_n()`` (batched
  GPU timesteps) → colour-map field to surface → blit → HUD.
- **Probes**: ``Probe`` objects accumulate ``(time, pressure)``.
  A ``_ProbePlotRenderer`` runs Matplotlib in a background thread so
  plot rendering never blocks the UI.
- **Layout**: grid left, probe panel right, HUD bar bottom.
  Dynamically recalculated on window resize (``VIDEORESIZE``).
- **Performance**: ``step_n()`` keeps the GPU busy for N steps with
  zero intermediate GPU→CPU transfers.  Probe values are subsampled
  (``probe_stride``) to avoid per-step GPU round-trips.
"""
from __future__ import annotations

import argparse
import sys
import threading
import time as _time
from pathlib import Path

import numpy as np
import pygame
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mpl_cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

from pressure_transfer_ca import (
    PressureCASimConfig,
    WaveStepper,
    _DiskFrameReader,
    auto_resolve_grid,
)

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# ── Config loading ─────────────────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


def _parse_range_list(raw) -> tuple:
    if not raw:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    out = []
    for tok in raw:
        tok = str(tok).strip()
        if not tok:
            continue
        s, e = tok.split(":", 1)
        out.append((int(s), int(e)))
    return tuple(out)


def config_to_sim(cfg: dict) -> PressureCASimConfig:
    ssp_depths = tuple(_get(cfg, "ssp", "depths", default=[0, 15, 35, 60, 90]))
    ssp_speeds = tuple(_get(cfg, "ssp", "speeds", default=[1535, 1518, 1492, 1503, 1520]))
    freq = _get(cfg, "source", "frequency", default=24000)

    stencil_order = _get(cfg, "grid", "stencil_order", default=2)
    grid = auto_resolve_grid(
        frequency_hz=freq,
        c_min=min(ssp_speeds),
        c_max=max(ssp_speeds),
        width_m=_get(cfg, "grid", "width_m", default=None),
        height_m=_get(cfg, "grid", "height_m", default=None),
        dx=_get(cfg, "grid", "dx", default=0),
        dy=_get(cfg, "grid", "dy", default=0),
        dt=_get(cfg, "time", "dt", default=0),
        nx=_get(cfg, "grid", "nx", default=0),
        ny=_get(cfg, "grid", "ny", default=0),
        cells_per_wavelength=_get(cfg, "grid", "cells_per_wavelength", default=0),
        stencil_order=stencil_order,
    )
    nx, ny = grid["nx"], grid["ny"]
    dx, dy, dt = grid["dx"], grid["dy"], grid["dt"]

    duration = _get(cfg, "time", "duration", default=0)
    steps_raw = _get(cfg, "time", "steps", default=1800)
    if duration and duration > 0:
        steps = max(1, int(round(duration / dt)))
    else:
        steps = steps_raw

    source_ix = _get(cfg, "source", "ix", default=None)
    source_iy = _get(cfg, "source", "iy", default=None)
    src_x = _get(cfg, "source", "x", default=None)
    src_y = _get(cfg, "source", "y", default=None)
    if src_x is not None:
        source_ix = int(round(float(src_x) / dx))
    if src_y is not None:
        source_iy = int(round(float(src_y) / dy))
    if source_ix is None:
        source_ix = nx // 2
    if source_iy is None:
        source_iy = ny // 2
    source_ix = max(0, min(source_ix, nx - 1))
    source_iy = max(0, min(source_iy, ny - 1))

    return PressureCASimConfig(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, steps=steps,
        propagation_model=_get(cfg, "model", "type", default="wave"),
        wave_absorption_per_s=_get(cfg, "model", "absorption", default=0.05),
        stencil_order=stencil_order,
        boundary_reflect_top=_get(cfg, "boundary", "top", default=-0.98),
        boundary_reflect_bottom=_get(cfg, "boundary", "bottom", default=0.99),
        boundary_reflect_left=_get(cfg, "boundary", "left", default=0.99),
        boundary_reflect_right=_get(cfg, "boundary", "right", default=0.99),
        top_open_x_ranges=_parse_range_list(_get(cfg, "boundary", "top_open_x", default=[])),
        bottom_open_x_ranges=_parse_range_list(_get(cfg, "boundary", "bottom_open_x", default=[])),
        left_open_y_ranges=_parse_range_list(_get(cfg, "boundary", "left_open_y", default=[])),
        right_open_y_ranges=_parse_range_list(_get(cfg, "boundary", "right_open_y", default=[])),
        backend=_get(cfg, "backend", default="auto"),
        use_float32=_get(cfg, "use_float32", default=True),
        source_ix=source_ix,
        source_iy=source_iy,
        source_amplitude_pa=_get(cfg, "source", "amplitude", default=1400.0),
        source_frequency_hz=freq,
        ssp_depths_m=ssp_depths,
        ssp_speeds_mps=ssp_speeds,
        transfer_fraction=_get(cfg, "transfer", "fraction", default=0.42),
        damping=_get(cfg, "transfer", "damping", default=0.002),
        diagonal_interface_scale=_get(cfg, "transfer", "diagonal_interface_scale", default=0.35),
        overpressure_only=_get(cfg, "transfer", "overpressure_only", default=False),
        refraction_strength=_get(cfg, "transfer", "refraction_strength", default=0.45),
        use_impedance_interface=_get(cfg, "transfer", "use_impedance_interface", default=True),
        frame_stride=1,
    )

# ── Colormap LUT (real matplotlib coolwarm) ───────────────────────────────

_N_LUT = 512
COOLWARM_LUT = (mpl_cm.coolwarm(np.linspace(0.0, 1.0, _N_LUT))[:, :3] * 255).astype(np.uint8)


def pressure_to_rgb(field: np.ndarray, vmax: float) -> np.ndarray:
    """Map pressure (ny, nx) -> (ny, nx, 3) uint8 using symmetric coolwarm."""
    vmax = max(vmax, 1e-30)
    norm = np.clip((field / vmax + 1.0) * 0.5, 0.0, 1.0)
    idx = (norm * (_N_LUT - 1)).astype(np.int32)
    return COOLWARM_LUT[idx]


def field_to_surface(field: np.ndarray, vmax: float) -> pygame.Surface:
    """Pressure field -> native-resolution pygame Surface (nx wide, ny tall)."""
    rgb = pressure_to_rgb(field, vmax)
    # pygame surfarray expects (width, height, 3) i.e. (nx, ny, 3)
    return pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))

# ── Probe: in-window panel rendered via matplotlib Agg ────────────────────

PROBE_COLORS = [
    (0, 200, 255),   # cyan
    (255, 100, 50),   # orange
    (100, 255, 100),  # green
    (255, 80, 255),   # magenta
    (255, 255, 80),   # yellow
    (80, 180, 255),   # light blue
    (255, 160, 160),  # pink
    (160, 255, 200),  # mint
]


class Probe:
    """Data collector for one probed cell."""

    def __init__(self, ix: int, iy: int, color_idx: int):
        self.ix = ix
        self.iy = iy
        self.color = PROBE_COLORS[color_idx % len(PROBE_COLORS)]
        self.mpl_color = tuple(c / 255.0 for c in self.color)
        self.times: list[float] = []
        self.pressures: list[float] = []

    def record(self, t: float, p_val: float):
        self.times.append(t)
        self.pressures.append(p_val)

    def clear(self):
        self.times.clear()
        self.pressures.clear()


PANEL_FRAC = 0.30
MIN_PANEL_W = 380
MAX_PROBES = 8
LEGEND_ROW_H = 22          # height of each clickable legend entry


def _source_sinusoid(cfg: PressureCASimConfig, t_max: float, n_pts: int = 2000):
    """Generate the clean source driver sinusoid at display resolution."""
    t = np.linspace(0, t_max, n_pts)
    s = cfg.source_amplitude_pa * np.sin(
        2.0 * np.pi * cfg.source_frequency_hz * t + cfg.source_phase_rad)
    if cfg.overpressure_only:
        s = np.maximum(s, 0.0)
    return t, s


def _render_probe_rgb(probe_data: list[dict], cfg: PressureCASimConfig,
                      plot_w: int, plot_h: int) -> np.ndarray | None:
    """Render probe plot to a numpy RGB array (thread-safe, no pygame calls).

    probe_data: list of dicts with keys 'times', 'pressures', 'mpl_color', 'ix', 'iy'.
    Returns (H, W, 3) uint8 array or None.
    """
    if not probe_data or plot_w < 50 or plot_h < 50:
        return None

    dpi = 100
    fig_w = plot_w / dpi
    fig_h = plot_h / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    t_max = 0.0
    has_data = False
    for pd in probe_data:
        times, pressures = pd["times"], pd["pressures"]
        if times:
            has_data = True
            t_max = max(t_max, times[-1])
            ax.plot(times, pressures, "-",
                    color=pd["mpl_color"], linewidth=1.0,
                    label=f"({pd['ix']},{pd['iy']}) d={pd['iy'] * cfg.dy:.0f}m",
                    zorder=5)

    if has_data and t_max > 0:
        ax2 = ax.twinx()
        t_src, s_src = _source_sinusoid(cfg, t_max)
        ax2.plot(t_src, s_src, "-", color="#ff6666", alpha=0.20,
                 linewidth=0.5, label="source driver")
        ax2.set_ylabel("Source (Pa)", fontsize=6, color="#ff6666")
        ax2.tick_params(axis="y", colors="#ff6666", labelsize=5)
        for spine in ax2.spines.values():
            spine.set_color("#555")
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.patch.set_visible(False)

    ax.set_title("Receiver pressure vs source driver", fontsize=8,
                 color="white", pad=3)
    ax.tick_params(colors="white", labelsize=6)
    for spine in ax.spines.values():
        spine.set_color("#555")
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xlabel("Time (s)", fontsize=7, color="white")
    ax.set_ylabel("Receiver (Pa)", fontsize=7, color="white")

    fig.tight_layout(pad=0.4)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = canvas.buffer_rgba()
    arr = np.asarray(buf)[:, :, :3].copy()
    plt.close(fig)
    return arr


class _ProbePlotRenderer:
    """Renders the probe plot in a background thread so it never blocks the main loop."""

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest_rgb: np.ndarray | None = None
        self._surface: pygame.Surface | None = None
        self._target_size: tuple[int, int] = (0, 0)

    def request_render(self, probe_data: list[dict], cfg: PressureCASimConfig,
                       plot_w: int, plot_h: int):
        """Kick off a background render if one isn't already running."""
        if self._thread is not None and self._thread.is_alive():
            return  # previous render still running, skip
        self._target_size = (plot_w, plot_h)
        self._thread = threading.Thread(
            target=self._worker, args=(probe_data, cfg, plot_w, plot_h),
            daemon=True)
        self._thread.start()

    def _worker(self, probe_data, cfg, plot_w, plot_h):
        rgb = _render_probe_rgb(probe_data, cfg, plot_w, plot_h)
        with self._lock:
            self._latest_rgb = rgb

    def get_surface(self, target_w: int, target_h: int) -> pygame.Surface | None:
        """Return the latest rendered surface (main thread only). Non-blocking."""
        with self._lock:
            rgb = self._latest_rgb
        if rgb is not None:
            try:
                surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
                self._surface = pygame.transform.scale(surf, (target_w, target_h))
            except Exception:
                pass
        return self._surface


# ── Main GUI ──────────────────────────────────────────────────────────────

HUD_FG = (255, 255, 255)
SOURCE_COLOR = (0, 0, 0)
GRID_COLOR = (40, 40, 40, 60)
OPEN_BOUNDARY_COLOR = (255, 80, 80)
BOUNDARY_COLOR = (200, 200, 200)
PANEL_BG = (30, 30, 30)
AXIS_COLOR = (200, 200, 200, 200)
AXIS_BG = (0, 0, 0, 140)
AXIS_MARGIN_LEFT = 48
AXIS_MARGIN_BOTTOM = 20
TICK_LEN = 5
HUD_H = 32


def _nice_tick_step(range_val, max_ticks=10):
    """Pick a human-friendly tick step for a given value range."""
    if range_val <= 0:
        return 1.0
    rough = range_val / max_ticks
    mag = 10 ** int(np.floor(np.log10(rough)))
    residual = rough / mag
    if residual <= 1.0:
        return mag
    elif residual <= 2.0:
        return 2 * mag
    elif residual <= 5.0:
        return 5 * mag
    return 10 * mag


def run_gui(sim_cfg: PressureCASimConfig, mode: str = "live",
            preloaded_frames: list | None = None,
            replay_meta: dict | None = None):
    pygame.init()
    caption = "Pressure CA — Interactive Viewer"
    if replay_meta:
        caption = f"Replay: {replay_meta['experiment']}"
    pygame.display.set_caption(caption)

    # Initial window size: fit the grid aspect ratio into screen
    info = pygame.display.Info()
    screen_w = min(info.current_w - 100, 1600)
    screen_h = min(info.current_h - 100, 1000)
    aspect = sim_cfg.nx / sim_cfg.ny
    init_grid_h = screen_h - HUD_H
    init_grid_w = min(int(init_grid_h * aspect), screen_w)
    init_grid_h = int(init_grid_w / aspect)
    init_w = init_grid_w
    init_h = init_grid_h + HUD_H

    screen = pygame.display.set_mode((init_w, init_h), pygame.RESIZABLE)
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 13)
    small_font = pygame.font.SysFont("monospace", 11)

    stepper = None
    backend_label = "replay"
    if preloaded_frames is not None:
        print(f"Replay mode: {len(preloaded_frames)} frames loaded")
    else:
        print(f"Initializing WaveStepper (backend={sim_cfg.backend})...")
        stepper = WaveStepper(sim_cfg)
        backend_label = stepper.backend_name
        print(f"Backend: {backend_label}")
    print(f"Grid: {sim_cfg.nx}x{sim_cfg.ny}")

    # ── Layout state (recalculated on resize) ──

    class Layout:
        """Mutable layout geometry, rebuilt on resize or panel toggle."""
        def __init__(self):
            self.win_w = init_w
            self.win_h = init_h
            self.panel_visible = False
            self.recalc()

        def recalc(self):
            total_w = self.win_w
            self.grid_h = self.win_h - HUD_H
            if self.panel_visible:
                self.panel_w = max(MIN_PANEL_W, int(total_w * PANEL_FRAC))
                self.grid_w = total_w - self.panel_w
            else:
                self.panel_w = 0
                self.grid_w = total_w
            self.grid_h = max(10, self.grid_h)
            self.grid_w = max(10, self.grid_w)
            self.cell_px_x = self.grid_w / sim_cfg.nx
            self.cell_px_y = self.grid_h / sim_cfg.ny
            self._grid_overlay = None

        def grid_overlay(self, viewport) -> pygame.Surface | None:
            """Build a grid overlay for the current viewport.

            Uses the viewport's visible cell range and draw_rect so grid
            lines track actual cell boundaries when zoomed/panned.
            """
            ox, oy, dw, dh = viewport.draw_rect()
            n_vis_x = viewport.world_w / sim_cfg.dx
            n_vis_y = viewport.world_h / sim_cfg.dy
            cpx = dw / max(1, n_vis_x)
            cpy = dh / max(1, n_vis_y)
            cell_px = min(cpx, cpy)
            if cell_px < 8:
                return None
            if self._grid_overlay is not None:
                return self._grid_overlay
            stride = 1
            for s in (1, 2, 5, 10, 20, 50):
                if cell_px * s >= 8:
                    stride = s
                    break
            surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
            gc = GRID_COLOR
            ix0 = int(viewport.world_x0 / sim_cfg.dx)
            iy0 = int(viewport.world_y0 / sim_cfg.dy)
            ix_start = ix0 - (ix0 % stride)
            iy_start = iy0 - (iy0 % stride)
            ix_end = min(sim_cfg.nx, int(np.ceil((viewport.world_x0 + viewport.world_w) / sim_cfg.dx)) + 1)
            iy_end = min(sim_cfg.ny, int(np.ceil((viewport.world_y0 + viewport.world_h) / sim_cfg.dy)) + 1)
            for ix in range(ix_start, ix_end + 1, stride):
                wx = ix * sim_cfg.dx
                px, _ = viewport.world_to_pixel(wx, 0)
                ipx = int(px)
                if 0 <= ipx < self.grid_w:
                    pygame.draw.line(surf, gc, (ipx, 0), (ipx, self.grid_h - 1), 1)
            for iy in range(iy_start, iy_end + 1, stride):
                wy = iy * sim_cfg.dy
                _, py = viewport.world_to_pixel(0, wy)
                ipy = int(py)
                if 0 <= ipy < self.grid_h:
                    pygame.draw.line(surf, gc, (0, ipy), (self.grid_w - 1, ipy), 1)
            self._grid_overlay = surf
            return surf

    layout = Layout()

    # ── Viewport (zoom & pan in world coordinates) ──

    class Viewport:
        """Tracks which rectangle of the world (in metres) is visible,
        with uniform scaling so the physical aspect ratio is preserved."""
        def __init__(self):
            self.reset()

        def reset(self):
            self.world_x0 = 0.0
            self.world_y0 = 0.0
            self.world_w = sim_cfg.nx * sim_cfg.dx
            self.world_h = sim_cfg.ny * sim_cfg.dy

        # -- draw rectangle: the pixel sub-rect within the grid area that
        #    preserves the physical aspect ratio (letterbox / pillarbox) --

        def draw_rect(self):
            """Return (ox, oy, dw, dh) — the pixel rectangle inside the grid
            area that maps to the current world window with uniform scaling."""
            gw, gh = layout.grid_w, layout.grid_h
            if self.world_w <= 0 or self.world_h <= 0:
                return 0, 0, gw, gh
            world_ar = self.world_w / self.world_h
            screen_ar = gw / gh
            if world_ar > screen_ar:
                dw = gw
                dh = int(gw / world_ar)
                ox = 0
                oy = (gh - dh) // 2
            else:
                dh = gh
                dw = int(gh * world_ar)
                ox = (gw - dw) // 2
                oy = 0
            return ox, oy, max(1, dw), max(1, dh)

        def zoom(self, factor, px, py):
            """Zoom by *factor* centred on screen pixel (px, py)."""
            wx, wy = self.pixel_to_world(px, py)
            full_w = sim_cfg.nx * sim_cfg.dx
            full_h = sim_cfg.ny * sim_cfg.dy
            new_w = max(sim_cfg.dx * 4, min(full_w, self.world_w / factor))
            new_h = max(sim_cfg.dy * 4, min(full_h, self.world_h / factor))
            # Lock aspect ratio to the world window's current ratio
            ar = self.world_w / self.world_h if self.world_h > 0 else 1.0
            if new_w / new_h > ar:
                new_h = new_w / ar
            else:
                new_w = new_h * ar
            new_w = min(new_w, full_w)
            new_h = min(new_h, full_h)
            ox, oy, dw, dh = self.draw_rect()
            frac_x = (px - ox) / dw if dw else 0.5
            frac_y = (py - oy) / dh if dh else 0.5
            frac_x = max(0.0, min(1.0, frac_x))
            frac_y = max(0.0, min(1.0, frac_y))
            self.world_x0 = wx - frac_x * new_w
            self.world_y0 = wy - frac_y * new_h
            self.world_w = new_w
            self.world_h = new_h
            self._clamp()
            layout._grid_overlay = None

        def pan(self, dx_px, dy_px):
            """Pan by a screen-pixel delta."""
            _, _, dw, dh = self.draw_rect()
            self.world_x0 -= dx_px / dw * self.world_w if dw else 0
            self.world_y0 -= dy_px / dh * self.world_h if dh else 0
            self._clamp()
            layout._grid_overlay = None

        def _clamp(self):
            full_w = sim_cfg.nx * sim_cfg.dx
            full_h = sim_cfg.ny * sim_cfg.dy
            self.world_x0 = max(0.0, min(self.world_x0, full_w - self.world_w))
            self.world_y0 = max(0.0, min(self.world_y0, full_h - self.world_h))

        def world_to_pixel(self, wx, wy):
            ox, oy, dw, dh = self.draw_rect()
            px = ox + (wx - self.world_x0) / self.world_w * dw
            py = oy + (wy - self.world_y0) / self.world_h * dh
            return px, py

        def pixel_to_world(self, px, py):
            ox, oy, dw, dh = self.draw_rect()
            wx = self.world_x0 + ((px - ox) / dw) * self.world_w
            wy = self.world_y0 + ((py - oy) / dh) * self.world_h
            return wx, wy

        def cell_slice(self):
            """Return (ix0, iy0, ix1, iy1) — the cell index range visible."""
            ix0 = max(0, int(self.world_x0 / sim_cfg.dx))
            iy0 = max(0, int(self.world_y0 / sim_cfg.dy))
            ix1 = min(sim_cfg.nx, int(np.ceil((self.world_x0 + self.world_w) / sim_cfg.dx)))
            iy1 = min(sim_cfg.ny, int(np.ceil((self.world_y0 + self.world_h) / sim_cfg.dy)))
            return ix0, iy0, ix1, iy1

        @property
        def zoom_level(self):
            full_w = sim_cfg.nx * sim_cfg.dx
            return full_w / self.world_w

    vp = Viewport()
    panning = False
    pan_last = (0, 0)

    # ── Simulation / display state ──

    playing = False
    TARGET_FPS = 30
    FRAME_BUDGET_S = 1.0 / TARGET_FPS
    steps_per_frame = 10
    spf_auto = True             # auto-tune spf to maintain TARGET_FPS
    speed_level = 0             # +/- keys shift this; each level doubles spf target
    sec_per_step = 0.001        # rolling estimate, updated each frame
    steps_per_sec_smooth = 0.0
    vmax_display = 1e-12
    probes: list[Probe] = []
    color_counter = 0
    frame_history: list[np.ndarray] = []   # only used in playback mode
    playback_idx = 0
    probe_dirty = True
    probe_redraw_counter = 0
    PROBE_REDRAW_INTERVAL = 20
    plot_renderer = _ProbePlotRenderer()

    # ── Helper functions ──

    def on_resize(new_w, new_h):
        nonlocal screen
        layout.win_w = max(200, new_w)
        layout.win_h = max(100, new_h)
        layout.recalc()
        screen = pygame.display.set_mode((layout.win_w, layout.win_h), pygame.RESIZABLE)

    def toggle_panel(show: bool):
        nonlocal probe_dirty
        layout.panel_visible = show
        layout.recalc()
        probe_dirty = True

    def pixel_to_cell(mx, my):
        if mx >= layout.grid_w or my >= layout.grid_h:
            return None
        wx, wy = vp.pixel_to_world(mx, my)
        ix = int(wx / sim_cfg.dx)
        iy = int(wy / sim_cfg.dy)
        if 0 <= ix < sim_cfg.nx and 0 <= iy < sim_cfg.ny:
            return ix, iy
        return None

    def cell_to_pixel(ix, iy):
        wx = (ix + 0.5) * sim_cfg.dx
        wy = (iy + 0.5) * sim_cfg.dy
        px, py = vp.world_to_pixel(wx, wy)
        return int(px), int(py)

    def fmt_pos(ix, iy):
        """Format cell position as physical coordinates."""
        return f"({ix * sim_cfg.dx:.1f}, {iy * sim_cfg.dy:.1f})m"

    def add_probe(ix, iy):
        nonlocal color_counter, probe_dirty
        for p in probes:
            if p.ix == ix and p.iy == iy:
                return
        if len(probes) >= MAX_PROBES:
            print(f"Max {MAX_PROBES} probes. Remove one first (right-click).")
            return
        p = Probe(ix, iy, color_counter)
        color_counter += 1
        if mode == "playback":
            for i, fld in enumerate(frame_history):
                if replay_meta and "frame_times" in replay_meta and i < len(replay_meta["frame_times"]):
                    t_frame = float(replay_meta["frame_times"][i])
                else:
                    t_frame = i * sim_cfg.dt
                p.record(t_frame, float(fld[iy, ix]))
        probes.append(p)
        probe_dirty = True
        if not layout.panel_visible:
            toggle_panel(True)
        print(f"Probe added at {fmt_pos(ix, iy)}")

    # legend_rects: list of (pygame.Rect in screen coords, probe index)
    legend_rects: list[tuple[pygame.Rect, int]] = []

    def remove_probe(idx):
        nonlocal probe_dirty
        removed = probes.pop(idx)
        probe_dirty = True
        print(f"Removed probe {fmt_pos(removed.ix, removed.iy)}")
        if not probes:
            toggle_panel(False)

    def handle_click_remove(mx, my):
        """Left-click on a legend entry or right-click on a grid marker removes a probe."""
        # Check legend entries first (panel area)
        for rect, pidx in legend_rects:
            if rect.collidepoint(mx, my) and pidx < len(probes):
                remove_probe(pidx)
                return True
        # Right-click on grid marker
        cell = pixel_to_cell(mx, my)
        if cell:
            ix, iy = cell
            for i, p in enumerate(probes):
                if p.ix == ix and p.iy == iy:
                    remove_probe(i)
                    return True
        return False

    # ── Rendering ──

    def _downsample_field(field, target_w, target_h):
        """Downsample a (ny, nx) array to roughly (target_h, target_w) via block mean."""
        ny, nx = field.shape
        if nx <= target_w and ny <= target_h:
            return field
        sy = max(1, ny // target_h)
        sx = max(1, nx // target_w)
        # Trim to exact multiple of block size, then reshape + mean
        trimmed = field[:ny - ny % sy, :nx - nx % sx]
        return trimmed.reshape(trimmed.shape[0] // sy, sy,
                               trimmed.shape[1] // sx, sx).mean(axis=(1, 3))

    def render_field(p_field):
        tw, th = layout.grid_w, layout.grid_h
        # Crop to the viewport's visible cell range
        ix0, iy0, ix1, iy1 = vp.cell_slice()
        visible = p_field[iy0:iy1, ix0:ix1]
        ds = _downsample_field(visible, tw, th)
        surf = field_to_surface(ds, vmax_display)
        sw, sh = surf.get_size()
        if sw > tw or sh > th:
            scaled = pygame.transform.smoothscale(surf, (tw, th))
        elif sw != tw or sh != th:
            scaled = pygame.transform.scale(surf, (tw, th))
        else:
            scaled = surf
        screen.blit(scaled, (0, 0))

        overlay = layout.grid_overlay(vp)
        if overlay:
            screen.blit(overlay, (0, 0))

        # Boundaries — only draw edges that are in the viewport
        lw = 3
        c = sim_cfg
        gw, gh = layout.grid_w, layout.grid_h
        top_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_top == 0.0 else BOUNDARY_COLOR
        bot_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_bottom == 0.0 else BOUNDARY_COLOR
        lft_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_left == 0.0 else BOUNDARY_COLOR
        rgt_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_right == 0.0 else BOUNDARY_COLOR
        full_w = sim_cfg.nx * sim_cfg.dx
        full_h = sim_cfg.ny * sim_cfg.dy
        _, top_py = vp.world_to_pixel(0, 0)
        _, bot_py = vp.world_to_pixel(0, full_h)
        lft_px, _ = vp.world_to_pixel(0, 0)
        rgt_px, _ = vp.world_to_pixel(full_w, 0)
        if 0 <= top_py <= gh:
            pygame.draw.line(screen, top_c, (0, int(top_py)), (gw - 1, int(top_py)), lw)
        if 0 <= bot_py <= gh:
            pygame.draw.line(screen, bot_c, (0, int(bot_py)), (gw - 1, int(bot_py)), lw)
        if 0 <= lft_px <= gw:
            pygame.draw.line(screen, lft_c, (int(lft_px), 0), (int(lft_px), gh - 1), lw)
        if 0 <= rgt_px <= gw:
            pygame.draw.line(screen, rgt_c, (int(rgt_px), 0), (int(rgt_px), gh - 1), lw)

        # Source marker
        sx, sy = cell_to_pixel(sim_cfg.source_ix, sim_cfg.source_iy)
        if 0 <= sx < gw and 0 <= sy < gh:
            r = max(3, int(min(layout.cell_px_x, layout.cell_px_y) * vp.zoom_level / 3))
            r = max(3, min(r, 20))
            pygame.draw.circle(screen, SOURCE_COLOR, (sx, sy), r)
            pygame.draw.circle(screen, (255, 255, 255), (sx, sy), r, 1)

        # Probe markers
        for p in probes:
            cx, cy = cell_to_pixel(p.ix, p.iy)
            if 0 <= cx < gw and 0 <= cy < gh:
                pr = max(4, int(min(layout.cell_px_x, layout.cell_px_y) * vp.zoom_level / 2))
                pr = max(4, min(pr, 20))
                pygame.draw.circle(screen, p.color, (cx, cy), pr, 2)
                label = small_font.render(fmt_pos(p.ix, p.iy), True, p.color)
                lx = min(cx + pr + 2, gw - label.get_width() - 2)
                ly = max(cy - label.get_height(), 2)
                screen.blit(label, (lx, ly))

    def render_axes():
        """Draw x/y coordinate axes with tick marks and labels (metres)."""
        gw, gh = layout.grid_w, layout.grid_h
        ax_font = small_font

        # Semi-transparent background strips for readability
        left_bg = pygame.Surface((AXIS_MARGIN_LEFT, gh), pygame.SRCALPHA)
        left_bg.fill(AXIS_BG)
        screen.blit(left_bg, (0, 0))
        bot_bg = pygame.Surface((gw, AXIS_MARGIN_BOTTOM), pygame.SRCALPHA)
        bot_bg.fill(AXIS_BG)
        screen.blit(bot_bg, (0, gh - AXIS_MARGIN_BOTTOM))

        # X-axis (bottom)
        x_step = _nice_tick_step(vp.world_w, max_ticks=max(3, gw // 80))
        x_start = np.ceil(vp.world_x0 / x_step) * x_step
        x = x_start
        while x <= vp.world_x0 + vp.world_w:
            px, _ = vp.world_to_pixel(x, 0)
            ipx = int(px)
            if AXIS_MARGIN_LEFT <= ipx < gw:
                ty = gh - AXIS_MARGIN_BOTTOM
                pygame.draw.line(screen, AXIS_COLOR[:3], (ipx, ty), (ipx, ty + TICK_LEN), 1)
                if x_step >= 1.0:
                    lbl = f"{x:.0f}"
                elif x_step >= 0.1:
                    lbl = f"{x:.1f}"
                else:
                    lbl = f"{x:.2f}"
                ts = ax_font.render(lbl, True, AXIS_COLOR[:3])
                screen.blit(ts, (ipx - ts.get_width() // 2, ty + TICK_LEN + 1))
            x += x_step

        # Y-axis (left)
        y_step = _nice_tick_step(vp.world_h, max_ticks=max(3, gh // 40))
        y_start = np.ceil(vp.world_y0 / y_step) * y_step
        y = y_start
        while y <= vp.world_y0 + vp.world_h:
            _, py = vp.world_to_pixel(0, y)
            ipy = int(py)
            if 0 <= ipy < gh - AXIS_MARGIN_BOTTOM:
                pygame.draw.line(screen, AXIS_COLOR[:3],
                                 (AXIS_MARGIN_LEFT - TICK_LEN, ipy),
                                 (AXIS_MARGIN_LEFT, ipy), 1)
                if y_step >= 1.0:
                    lbl = f"{y:.0f}"
                elif y_step >= 0.1:
                    lbl = f"{y:.1f}"
                else:
                    lbl = f"{y:.2f}"
                ts = ax_font.render(lbl, True, AXIS_COLOR[:3])
                screen.blit(ts, (AXIS_MARGIN_LEFT - TICK_LEN - ts.get_width() - 2,
                                 ipy - ts.get_height() // 2))
            y += y_step

        # Axis labels
        x_label = ax_font.render("x (m)", True, AXIS_COLOR[:3])
        screen.blit(x_label, (gw // 2 - x_label.get_width() // 2,
                               gh - x_label.get_height()))
        y_label = ax_font.render("y (m)", True, AXIS_COLOR[:3])
        screen.blit(y_label, (2, 2))

    def render_panel():
        nonlocal probe_dirty
        if not layout.panel_visible:
            legend_rects.clear()
            return
        px = layout.grid_w
        panel_rect = pygame.Rect(px, 0, layout.panel_w, layout.grid_h)
        pygame.draw.rect(screen, PANEL_BG, panel_rect)

        # ── Clickable legend bar at top of panel ──
        legend_rects.clear()
        legend_y = 4
        for i, p in enumerate(probes):
            label_text = f"  \u25a0 {fmt_pos(p.ix, p.iy)} d={p.iy * sim_cfg.dy:.1f}m  \u2715 "
            label_surf = small_font.render(label_text, True, p.color)
            lw = label_surf.get_width()
            lh = LEGEND_ROW_H
            rect = pygame.Rect(px + 4, legend_y, lw + 4, lh)
            mx, my = pygame.mouse.get_pos()
            if rect.collidepoint(mx, my):
                pygame.draw.rect(screen, (60, 60, 60), rect, border_radius=3)
            screen.blit(label_surf, (px + 6, legend_y + 2))
            legend_rects.append((rect, i))
            legend_y += lh

        legend_total_h = legend_y + 2
        plot_h = layout.grid_h - legend_total_h

        # ── Kick off background render if dirty; blit latest available surface ──
        if probe_dirty:
            snap = [{"times": list(p.times), "pressures": list(p.pressures),
                     "mpl_color": p.mpl_color, "ix": p.ix, "iy": p.iy}
                    for p in probes]
            plot_renderer.request_render(snap, sim_cfg,
                                         layout.panel_w, max(60, plot_h))
            probe_dirty = False

        surf = plot_renderer.get_surface(layout.panel_w, max(60, plot_h))
        if surf:
            screen.blit(surf, (px, legend_total_h))

    def render_hud(step_num, t, fps_val, backend, mode_str, spf, sps=0.0):
        rect = pygame.Rect(0, layout.grid_h, layout.win_w, HUD_H)
        pygame.draw.rect(screen, (30, 30, 30), rect)
        state = "PLAY" if playing else "PAUSE"
        zoom_str = f"{vp.zoom_level:.1f}x" if vp.zoom_level > 1.01 else "1x"
        # Cursor world position
        mx, my = pygame.mouse.get_pos()
        if mx < layout.grid_w and my < layout.grid_h:
            cwx, cwy = vp.pixel_to_world(mx, my)
            cursor_str = f"({cwx:.1f},{cwy:.1f})m"
        else:
            cursor_str = ""
        if replay_meta:
            n_total = replay_meta["n_frames"]
            txt = (f" frame {step_num}/{n_total}  t={t:.5f}s  |  {fps_val:.0f}fps  |  "
                   f"spf={spf}  |  speed {speed_level:+d}  |  "
                   f"zoom {zoom_str}  |  {cursor_str}  |  "
                   f"REPLAY [{state}]  |  vmax={vmax_display:.2e}")
        else:
            sps_str = f"{sps/1000:.1f}k" if sps >= 1000 else f"{sps:.0f}"
            txt = (f" step {step_num:>6d}  t={t:.5f}s  |  {fps_val:.0f}fps  |  "
                   f"{sps_str} sps  spf={spf}  speed {speed_level:+d}  |  "
                   f"zoom {zoom_str}  |  {cursor_str}  |  "
                   f"{backend} [{state}]  |  vmax={vmax_display:.2e}")
        surf = font.render(txt, True, HUD_FG)
        screen.blit(surf, (4, layout.grid_h + 7))

    # ── Pre-compute or load frames (playback mode) ──

    if mode == "playback":
        if preloaded_frames is not None:
            frame_history = preloaded_frames
            print(f"Loaded {len(frame_history)} replay frames.")
        elif stepper is not None:
            print(f"Pre-computing {sim_cfg.steps} steps...")
            for s in range(sim_cfg.steps):
                p = stepper.step()
                frame_history.append(p)
                if (s + 1) % 500 == 0:
                    print(f"  {s + 1}/{sim_cfg.steps}")
            print("Pre-compute done. Starting playback.")
        playback_idx = 0

    if mode == "playback" and frame_history:
        current_field = frame_history[0]
    else:
        current_field = np.zeros((sim_cfg.ny, sim_cfg.nx), dtype=np.float32)

    running = True

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.VIDEORESIZE:
                on_resize(ev.w, ev.h)
            elif ev.type == pygame.KEYDOWN:
                if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif ev.key == pygame.K_SPACE:
                    playing = not playing
                elif ev.key == pygame.K_r:
                    if mode == "live":
                        stepper.reset()
                        for p in probes:
                            p.clear()
                        probe_dirty = True
                        vmax_display = 1e-12
                        current_field = np.zeros((sim_cfg.ny, sim_cfg.nx), dtype=np.float32)
                    else:
                        playback_idx = 0
                elif ev.key == pygame.K_HOME:
                    vp.reset()
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    speed_level = min(20, speed_level + 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed_level = max(-5, speed_level - 1)
                elif ev.key == pygame.K_RIGHT and not playing:
                    if mode == "live":
                        pc = [(p.ix, p.iy) for p in probes] if probes else None
                        current_field, pv, pt = stepper.step_n(1, pc)
                        if pv is not None:
                            for j, p in enumerate(probes):
                                p.record(float(pt[0]), float(pv[0, j]))
                        probe_dirty = True
                    elif mode == "playback" and playback_idx < len(frame_history) - 1:
                        playback_idx += 1
                        current_field = frame_history[playback_idx]
                elif ev.key == pygame.K_LEFT and not playing and mode == "playback":
                    if playback_idx > 0:
                        playback_idx -= 1
                        current_field = frame_history[playback_idx]
            elif ev.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if mx < layout.grid_w and my < layout.grid_h:
                    factor = 1.15 ** ev.y
                    vp.zoom(factor, mx, my)
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 2:  # middle click — start pan
                    panning = True
                    pan_last = ev.pos
                elif ev.button == 1:
                    if not handle_click_remove(*ev.pos):
                        cell = pixel_to_cell(*ev.pos)
                        if cell:
                            add_probe(*cell)
                elif ev.button == 3:
                    handle_click_remove(*ev.pos)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 2:
                    panning = False
            elif ev.type == pygame.MOUSEMOTION:
                if panning:
                    dx = ev.pos[0] - pan_last[0]
                    dy = ev.pos[1] - pan_last[1]
                    vp.pan(dx, dy)
                    pan_last = ev.pos

        # Advance simulation — auto-tune spf to stay within frame budget
        sim_steps_this_frame = 0
        t_sim_start = _time.perf_counter()
        if playing:
            if mode == "live":
                # Target spf: base 10 * 2^speed_level, clamped to [1, 500000]
                target_spf = max(1, min(500000, int(10 * (2.0 ** speed_level))))
                # Auto-tune: estimate how many steps we can afford in the budget
                if sec_per_step > 0:
                    affordable = max(1, int(FRAME_BUDGET_S * 0.7 / sec_per_step))
                else:
                    affordable = target_spf
                steps_per_frame = min(target_spf, affordable)

                probe_cells = [(p.ix, p.iy) for p in probes] if probes else None
                pstride = max(1, steps_per_frame // 200)
                current_field, pv, pt = stepper.step_n(
                    steps_per_frame, probe_cells, probe_stride=pstride)
                sim_steps_this_frame = steps_per_frame
                if pv is not None:
                    for row_t, row_v in zip(pt, pv):
                        for j, p in enumerate(probes):
                            p.record(float(row_t), float(row_v[j]))
                probe_dirty = True
            elif mode == "playback":
                playback_idx = min(playback_idx + steps_per_frame, len(frame_history) - 1)
                current_field = frame_history[playback_idx]
                if playback_idx >= len(frame_history) - 1:
                    playing = False
        t_sim_elapsed = _time.perf_counter() - t_sim_start
        if t_sim_elapsed > 0 and sim_steps_this_frame > 0:
            measured_sps = sim_steps_this_frame / t_sim_elapsed
            sec_per_step = 0.8 * sec_per_step + 0.2 * (t_sim_elapsed / sim_steps_this_frame)
            steps_per_sec_smooth = 0.9 * steps_per_sec_smooth + 0.1 * measured_sps

        # Dynamic color scaling: track field peak, decay slowly so colors stay stable
        field_max = float(np.max(np.abs(current_field)))
        if field_max > vmax_display:
            vmax_display = field_max
        else:
            vmax_display = max(field_max, vmax_display * 0.998)
        vmax_display = max(vmax_display, 1e-30)

        # Render
        render_field(current_field)
        render_axes()

        # Throttle probe plot re-renders (background thread handles the heavy work)
        probe_redraw_counter += 1
        if probe_redraw_counter >= PROBE_REDRAW_INTERVAL and playing:
            probe_redraw_counter = 0
            probe_dirty = True
        render_panel()

        if mode == "live":
            step_num = stepper.current_step
            t = stepper.current_time
        else:
            step_num = playback_idx
            if replay_meta and "frame_times" in replay_meta and playback_idx < len(replay_meta["frame_times"]):
                t = float(replay_meta["frame_times"][playback_idx])
            else:
                t = playback_idx * sim_cfg.dt

        fps_val = clock.get_fps()
        render_hud(step_num, t, fps_val, backend_label, mode,
                   steps_per_frame, steps_per_sec_smooth)
        pygame.display.flip()
        clock.tick(0)  # no cap — auto-tuning handles pacing

    pygame.quit()


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Real-time interactive pressure CA viewer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config", type=str, default=None,
                    help="YAML config file (default: configs/default.yaml)")
    p.add_argument("--mode", type=str, default="live", choices=["live", "playback"],
                    help="live = compute on-the-fly, playback = pre-compute then scrub")
    p.add_argument("--replay", type=str, default=None, metavar="EXPERIMENT_DIR",
                    help="Replay a saved experiment (loads frames.npz + config.yaml)")
    p.add_argument("--backend", type=str, default=None, choices=["auto", "cpu", "gpu"])
    return p.parse_args()


def _load_replay(replay_dir: Path):
    """Load frames from an experiment directory.

    Supports two storage formats:
    - ``frames.npz`` — compressed archive with metadata (legacy / small runs)
    - ``_frames_cache.bin`` — flat binary produced by ``_FrameSink`` disk mode

    Returns (sim_cfg, frame_source, replay_meta).  *frame_source* is an
    indexable sequence (list or ``_DiskFrameReader``).
    """
    cfg_path = replay_dir / "config.yaml"
    npz_path = replay_dir / "frames.npz"
    bin_path = replay_dir / "_frames_cache.bin"

    if not cfg_path.exists():
        print(f"ERROR: {cfg_path} not found in replay dir")
        sys.exit(1)

    cfg_dict = load_config(cfg_path)
    sim_cfg = config_to_sim(cfg_dict)

    # ── Try frames.npz first ──
    if npz_path.exists():
        data = np.load(npz_path)
        frames_arr = data["frames"]
        frame_times = data["frame_times"]
        ds_factor = int(data["ds_factor"]) if "ds_factor" in data else 1
        orig_nx = int(data["nx"]) if "nx" in data else sim_cfg.nx
        orig_ny = int(data["ny"]) if "ny" in data else sim_cfg.ny

        frame_list = [frames_arr[i] for i in range(frames_arr.shape[0])]

        stored_ny, stored_nx = frames_arr.shape[1], frames_arr.shape[2]
        if stored_nx != sim_cfg.nx or stored_ny != sim_cfg.ny:
            sim_cfg = PressureCASimConfig(**{
                **sim_cfg.__dict__,
                "nx": stored_nx, "ny": stored_ny,
                "dx": sim_cfg.dx * ds_factor, "dy": sim_cfg.dy * ds_factor,
                "source_ix": sim_cfg.source_ix // ds_factor,
                "source_iy": sim_cfg.source_iy // ds_factor,
            })

        meta = {
            "experiment": replay_dir.name,
            "n_frames": len(frame_list),
            "ds_factor": ds_factor,
            "orig_nx": orig_nx,
            "orig_ny": orig_ny,
            "frame_times": frame_times,
        }
        return sim_cfg, frame_list, meta

    # ── Fall back to _frames_cache.bin ──
    if bin_path.exists():
        import json
        meta_json_path = replay_dir / "_frames_cache_meta.json"
        if meta_json_path.exists():
            with open(meta_json_path) as f:
                fmeta = json.load(f)
            stored_ny = fmeta["ny"]
            stored_nx = fmeta["nx"]
            n_frames = fmeta["count"]
            ds_factor = fmeta.get("ds", 1)
        else:
            stored_ny, stored_nx = sim_cfg.ny, sim_cfg.nx
            ds_factor = 1
            file_bytes = bin_path.stat().st_size
            frame_nbytes = stored_ny * stored_nx * 4
            n_frames = file_bytes // frame_nbytes

        if n_frames == 0:
            print(f"ERROR: {bin_path} has no usable frames")
            sys.exit(1)

        reader = _DiskFrameReader(str(bin_path), n_frames,
                                  stored_ny, stored_nx, np.float32)

        if stored_nx != sim_cfg.nx or stored_ny != sim_cfg.ny:
            sim_cfg = PressureCASimConfig(**{
                **sim_cfg.__dict__,
                "nx": stored_nx, "ny": stored_ny,
                "dx": sim_cfg.dx * ds_factor, "dy": sim_cfg.dy * ds_factor,
                "source_ix": sim_cfg.source_ix // ds_factor,
                "source_iy": sim_cfg.source_iy // ds_factor,
            })

        stride = max(1, sim_cfg.frame_stride)
        frame_times = np.arange(n_frames) * sim_cfg.dt * stride
        meta = {
            "experiment": replay_dir.name,
            "n_frames": n_frames,
            "ds_factor": ds_factor,
            "orig_nx": sim_cfg.nx * ds_factor,
            "orig_ny": sim_cfg.ny * ds_factor,
            "frame_times": frame_times,
        }
        return sim_cfg, reader, meta

    print(f"ERROR: No frame data found in {replay_dir}. "
          f"Need frames.npz or _frames_cache.bin.")
    sys.exit(1)


def main():
    args = parse_args()

    replay_meta = None

    if args.replay:
        replay_dir = Path(args.replay)
        if not replay_dir.is_absolute():
            replay_dir = PROJECT_ROOT / args.replay
        if not replay_dir.exists():
            # Try as a subdirectory of experiments/
            alt = PROJECT_ROOT / "experiments" / args.replay
            if alt.exists():
                replay_dir = alt
            else:
                print(f"ERROR: replay dir not found: {replay_dir}")
                sys.exit(1)

        print(f"Loading replay: {replay_dir}")
        sim_cfg, frame_list, replay_meta = _load_replay(replay_dir)
        mode = "playback"
        ds = replay_meta['ds_factor']
        ds_note = f", downsampled {ds}x" if ds > 1 else ""
        print(f"Grid: {sim_cfg.nx}x{sim_cfg.ny} ({replay_meta['n_frames']} frames{ds_note})")
    else:
        config_path = Path(args.config) if args.config else DEFAULT_CONFIG
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        if not config_path.exists():
            print(f"ERROR: config not found: {config_path}")
            sys.exit(1)

        print(f"Loading config: {config_path}")
        cfg_dict = load_config(config_path)
        sim_cfg = config_to_sim(cfg_dict)
        frame_list = None
        mode = args.mode

    if args.backend:
        sim_cfg = PressureCASimConfig(**{**sim_cfg.__dict__, "backend": args.backend})

    print(f"Grid: {sim_cfg.nx}x{sim_cfg.ny}, dx={sim_cfg.dx}, dy={sim_cfg.dy}")
    print(f"dt={sim_cfg.dt:.6f}, steps={sim_cfg.steps}, mode={mode}")

    run_gui(sim_cfg, mode=mode, preloaded_frames=frame_list, replay_meta=replay_meta)


if __name__ == "__main__":
    main()
