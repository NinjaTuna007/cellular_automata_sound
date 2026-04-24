#!/usr/bin/env python3
"""Pygame real-time interactive viewer for the pressure-transfer CA.

Usage:
  python scripts/gui.py --config configs/no_side_reflection.yaml
  python scripts/gui.py --config configs/no_side_reflection.yaml --mode playback

Controls:
  Space        Play / pause
  Right arrow  Step forward (when paused)
  Left arrow   Step backward (playback mode, when paused)
  R            Reset to step 0
  +/=          More steps per frame (faster)
  -            Fewer steps per frame (slower)
  Click grid   Open probe (pressure time-series vs source)
  Right-click  Remove probe under cursor (on grid marker or panel)
  Esc / Q      Quit
"""
from __future__ import annotations

import argparse
import sys
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
    nx = _get(cfg, "grid", "nx", default=160)
    ny = _get(cfg, "grid", "ny", default=90)
    dx = _get(cfg, "grid", "dx", default=1.0)
    dy = _get(cfg, "grid", "dy", default=1.0)

    dt_raw = _get(cfg, "time", "dt", default=0)
    duration = _get(cfg, "time", "duration", default=0)
    steps_raw = _get(cfg, "time", "steps", default=1800)

    ssp_depths = tuple(_get(cfg, "ssp", "depths", default=[0, 15, 35, 60, 90]))
    ssp_speeds = tuple(_get(cfg, "ssp", "speeds", default=[1535, 1518, 1492, 1503, 1520]))
    cmax = max(ssp_speeds)

    dt_auto = 0.45 / (cmax * ((1.0 / dx**2 + 1.0 / dy**2) ** 0.5))
    dt = dt_raw if dt_raw and dt_raw > 0 else dt_auto

    if duration and duration > 0:
        steps = max(1, int(round(duration / dt)))
    else:
        steps = steps_raw

    source_ix = _get(cfg, "source", "ix", default=None)
    source_iy = _get(cfg, "source", "iy", default=None)
    if source_ix is None:
        source_ix = nx // 2
    if source_iy is None:
        source_iy = ny // 2

    return PressureCASimConfig(
        nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, steps=steps,
        propagation_model=_get(cfg, "model", "type", default="wave"),
        wave_absorption_per_s=_get(cfg, "model", "absorption", default=0.05),
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
        source_frequency_hz=_get(cfg, "source", "frequency", default=24000),
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


def render_probe_plot(probes: list[Probe], cfg: PressureCASimConfig,
                      plot_w: int, plot_h: int) -> pygame.Surface | None:
    """Render a single combined plot with all probes overlaid via matplotlib Agg."""
    if not probes or plot_w < 50 or plot_h < 50:
        return None

    dpi = 100
    fig_w = plot_w / dpi
    fig_h = plot_h / dpi

    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#2a2a2a")

    t_max = 0.0
    has_data = False
    for probe in probes:
        if probe.times:
            has_data = True
            t_max = max(t_max, probe.times[-1])
            ax.plot(probe.times, probe.pressures, "-",
                    color=probe.mpl_color, linewidth=1.0,
                    label=f"({probe.ix},{probe.iy}) d={probe.iy * cfg.dy:.0f}m",
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
    arr = np.asarray(buf)
    plt.close(fig)

    rgb = arr[:, :, :3].copy()
    surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
    return pygame.transform.scale(surf, (plot_w, plot_h))


# ── Main GUI ──────────────────────────────────────────────────────────────

HUD_FG = (255, 255, 255)
SOURCE_COLOR = (0, 0, 0)
GRID_COLOR = (40, 40, 40, 60)
OPEN_BOUNDARY_COLOR = (255, 80, 80)
BOUNDARY_COLOR = (200, 200, 200)
PANEL_BG = (30, 30, 30)
HUD_H = 32


def run_gui(sim_cfg: PressureCASimConfig, mode: str = "live"):
    pygame.init()
    pygame.display.set_caption("Pressure CA — Interactive Viewer")

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

    print(f"Initializing WaveStepper (backend={sim_cfg.backend})...")
    stepper = WaveStepper(sim_cfg)
    print(f"Backend: {stepper.backend_name}")
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

        def grid_overlay(self) -> pygame.Surface | None:
            """Lazy-built semi-transparent grid line overlay."""
            cpx = min(self.cell_px_x, self.cell_px_y)
            if cpx < 8:
                return None
            if self._grid_overlay is not None:
                return self._grid_overlay
            stride = 1
            for s in (1, 2, 5, 10, 20, 50):
                if cpx * s >= 8:
                    stride = s
                    break
            surf = pygame.Surface((self.grid_w, self.grid_h), pygame.SRCALPHA)
            gc = GRID_COLOR
            for ix in range(0, sim_cfg.nx + 1, stride):
                x = int(ix * self.cell_px_x)
                pygame.draw.line(surf, gc, (x, 0), (x, self.grid_h - 1), 1)
            for iy in range(0, sim_cfg.ny + 1, stride):
                y = int(iy * self.cell_px_y)
                pygame.draw.line(surf, gc, (0, y), (self.grid_w - 1, y), 1)
            self._grid_overlay = surf
            return surf

    layout = Layout()

    # ── Simulation / display state ──

    playing = False
    steps_per_frame = 1
    vmax_display = 1e-12        # tracks the running max with smooth decay
    VMAX_DECAY = 0.998          # per-frame multiplicative decay
    probes: list[Probe] = []
    color_counter = 0
    frame_history: list[np.ndarray] = []
    playback_idx = 0
    probe_surface: pygame.Surface | None = None
    probe_dirty = True
    probe_redraw_counter = 0
    PROBE_REDRAW_INTERVAL = 15

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
        ix = int(mx / layout.cell_px_x)
        iy = int(my / layout.cell_px_y)
        if 0 <= ix < sim_cfg.nx and 0 <= iy < sim_cfg.ny:
            return ix, iy
        return None

    def cell_to_pixel(ix, iy):
        return (int((ix + 0.5) * layout.cell_px_x),
                int((iy + 0.5) * layout.cell_px_y))

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
        for i, fld in enumerate(frame_history):
            p.record(i * sim_cfg.dt, float(fld[iy, ix]))
        probes.append(p)
        probe_dirty = True
        if not layout.panel_visible:
            toggle_panel(True)
        print(f"Probe added at ({ix}, {iy})")

    # legend_rects: list of (pygame.Rect in screen coords, probe index)
    legend_rects: list[tuple[pygame.Rect, int]] = []

    def remove_probe(idx):
        nonlocal probe_dirty
        removed = probes.pop(idx)
        probe_dirty = True
        print(f"Removed probe ({removed.ix}, {removed.iy})")
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

    def render_field(p_field):
        native_surf = field_to_surface(p_field, vmax_display)
        scaled = pygame.transform.scale(native_surf, (layout.grid_w, layout.grid_h))
        screen.blit(scaled, (0, 0))

        overlay = layout.grid_overlay()
        if overlay:
            screen.blit(overlay, (0, 0))

        # Boundaries
        lw = 3
        c = sim_cfg
        top_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_top == 0.0 else BOUNDARY_COLOR
        bot_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_bottom == 0.0 else BOUNDARY_COLOR
        lft_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_left == 0.0 else BOUNDARY_COLOR
        rgt_c = OPEN_BOUNDARY_COLOR if c.boundary_reflect_right == 0.0 else BOUNDARY_COLOR
        gw, gh = layout.grid_w, layout.grid_h
        pygame.draw.line(screen, top_c, (0, 0), (gw - 1, 0), lw)
        pygame.draw.line(screen, bot_c, (0, gh - 1), (gw - 1, gh - 1), lw)
        pygame.draw.line(screen, lft_c, (0, 0), (0, gh - 1), lw)
        pygame.draw.line(screen, rgt_c, (gw - 1, 0), (gw - 1, gh - 1), lw)

        # Source marker
        sx, sy = cell_to_pixel(sim_cfg.source_ix, sim_cfg.source_iy)
        r = max(3, int(min(layout.cell_px_x, layout.cell_px_y) / 3))
        pygame.draw.circle(screen, SOURCE_COLOR, (sx, sy), r)
        pygame.draw.circle(screen, (255, 255, 255), (sx, sy), r, 1)

        # Probe markers
        for p in probes:
            cx, cy = cell_to_pixel(p.ix, p.iy)
            pr = max(4, int(min(layout.cell_px_x, layout.cell_px_y) / 2))
            pygame.draw.circle(screen, p.color, (cx, cy), pr, 2)
            label = small_font.render(f"({p.ix},{p.iy})", True, p.color)
            lx = min(cx + pr + 2, gw - label.get_width() - 2)
            ly = max(cy - label.get_height(), 2)
            screen.blit(label, (lx, ly))

    def render_panel():
        nonlocal probe_surface, probe_dirty
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
            label_text = f"  ■ ({p.ix},{p.iy}) d={p.iy * sim_cfg.dy:.0f}m  ✕ "
            label_surf = small_font.render(label_text, True, p.color)
            lw = label_surf.get_width()
            lh = LEGEND_ROW_H
            rect = pygame.Rect(px + 4, legend_y, lw + 4, lh)
            # Hover highlight
            mx, my = pygame.mouse.get_pos()
            if rect.collidepoint(mx, my):
                pygame.draw.rect(screen, (60, 60, 60), rect, border_radius=3)
            screen.blit(label_surf, (px + 6, legend_y + 2))
            legend_rects.append((rect, i))
            legend_y += lh

        legend_total_h = legend_y + 2
        plot_h = layout.grid_h - legend_total_h

        # ── Combined matplotlib plot below legend ──
        if probe_dirty or probe_surface is None:
            probe_surface = render_probe_plot(probes, sim_cfg,
                                              layout.panel_w, max(60, plot_h))
            probe_dirty = False

        if probe_surface:
            screen.blit(probe_surface, (px, legend_total_h))

    def render_hud(step_num, t, fps_val, backend, mode_str, spf):
        rect = pygame.Rect(0, layout.grid_h, layout.win_w, HUD_H)
        pygame.draw.rect(screen, (30, 30, 30), rect)
        state = "PLAY" if playing else "PAUSE"
        txt = (f" step {step_num:>6d}  t={t:.5f}s  |  {fps_val:.0f}fps  |  "
               f"{backend}  |  {mode_str}  |  spf={spf}  |  [{state}]  |  "
               f"vmax={vmax_display:.2e}  |  "
               f"click=probe  rclick=remove  space=play  +/-=speed  R=reset")
        surf = font.render(txt, True, HUD_FG)
        screen.blit(surf, (4, layout.grid_h + 7))

    # ── Pre-compute (playback mode) ──

    if mode == "playback":
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
                        frame_history.clear()
                        for p in probes:
                            p.clear()
                        probe_dirty = True
                        vmax_display = 1e-12
                        current_field = np.zeros((sim_cfg.ny, sim_cfg.nx), dtype=np.float32)
                    else:
                        playback_idx = 0
                elif ev.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    steps_per_frame = min(100, steps_per_frame + 1)
                elif ev.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    steps_per_frame = max(1, steps_per_frame - 1)
                elif ev.key == pygame.K_RIGHT and not playing:
                    if mode == "live":
                        current_field = stepper.step()
                        frame_history.append(current_field)
                        for p in probes:
                            p.record(stepper.current_time, float(current_field[p.iy, p.ix]))
                        probe_dirty = True
                    elif mode == "playback" and playback_idx < len(frame_history) - 1:
                        playback_idx += 1
                        current_field = frame_history[playback_idx]
                elif ev.key == pygame.K_LEFT and not playing and mode == "playback":
                    if playback_idx > 0:
                        playback_idx -= 1
                        current_field = frame_history[playback_idx]
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    if not handle_click_remove(*ev.pos):
                        cell = pixel_to_cell(*ev.pos)
                        if cell:
                            add_probe(*cell)
                elif ev.button == 3:
                    handle_click_remove(*ev.pos)

        # Advance simulation
        if playing:
            if mode == "live":
                for _ in range(steps_per_frame):
                    current_field = stepper.step()
                    frame_history.append(current_field)
                    for p in probes:
                        p.record(stepper.current_time, float(current_field[p.iy, p.ix]))
                probe_dirty = True
            elif mode == "playback":
                playback_idx = min(playback_idx + steps_per_frame, len(frame_history) - 1)
                current_field = frame_history[playback_idx]
                if playback_idx >= len(frame_history) - 1:
                    playing = False

        # Dynamic color scaling: track field peak, decay slowly so colors stay stable
        field_max = float(np.max(np.abs(current_field)))
        if field_max > vmax_display:
            vmax_display = field_max
        else:
            vmax_display = max(field_max, vmax_display * VMAX_DECAY)
        vmax_display = max(vmax_display, 1e-30)

        # Render
        render_field(current_field)

        probe_redraw_counter += 1
        if probe_redraw_counter >= PROBE_REDRAW_INTERVAL:
            probe_redraw_counter = 0
            if probes and probe_dirty:
                probe_dirty = True
        render_panel()

        if mode == "live":
            step_num = stepper.current_step
            t = stepper.current_time
        else:
            step_num = playback_idx
            t = playback_idx * sim_cfg.dt

        fps_val = clock.get_fps()
        render_hud(step_num, t, fps_val, stepper.backend_name, mode, steps_per_frame)
        pygame.display.flip()
        clock.tick(60)

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
    p.add_argument("--backend", type=str, default=None, choices=["auto", "cpu", "gpu"])
    return p.parse_args()


def main():
    args = parse_args()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    if not config_path.exists():
        print(f"ERROR: config not found: {config_path}")
        sys.exit(1)

    print(f"Loading config: {config_path}")
    cfg_dict = load_config(config_path)
    sim_cfg = config_to_sim(cfg_dict)

    if args.backend:
        sim_cfg = PressureCASimConfig(**{**sim_cfg.__dict__, "backend": args.backend})

    print(f"Grid: {sim_cfg.nx}x{sim_cfg.ny}, dx={sim_cfg.dx}, dy={sim_cfg.dy}")
    print(f"dt={sim_cfg.dt:.6f}, steps={sim_cfg.steps}, mode={args.mode}")

    run_gui(sim_cfg, mode=args.mode)


if __name__ == "__main__":
    main()
