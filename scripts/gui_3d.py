#!/usr/bin/env python3
"""3D acoustic wave simulation — real-time OpenGL volume rendering.

Uses vispy for GPU-accelerated volume rendering with a turntable camera.
Supports live simulation, pre-computed replay, probe placement, and
interactive slice planes.

Controls
--------
Left-drag       Rotate camera
Right-drag      Pan
Scroll          Zoom
Left-click      Place pressure probe at cursor position
P               Toggle pause
+/-             Increase/decrease steps per frame
C               Cycle colormap (coolwarm → hot → grays)
M               Cycle render method (mip → translucent → iso)
X/Y/Z           Toggle slice plane along that axis
[/]             Move active slice plane backward/forward
R               Reset camera
Backspace       Remove last probe
Q / Escape      Quit

Usage
-----
Live simulation::

    python scripts/gui_3d.py --config configs/room_3d.yaml

Replay::

    python scripts/gui_3d.py --replay experiments/20260427_143000
"""
from __future__ import annotations

import argparse
import json
import sys
import time as _time
from collections import deque
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from wave_3d import (
    WaveConfig3D,
    WaveStepper3D,
    _DiskFrameReader3D,
    auto_resolve_grid_3d,
    build_sound_speed_grid_3d,
    check_cfl_3d,
)

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

import vispy
vispy.use(app="pyqt5", gl="gl2")

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore

from vispy import app, scene
from vispy.color import get_colormap
from vispy.scene import visuals

# ─── Config loading ─────────────────────────────────────────────────────────


def _get(cfg: dict, *keys, default=None):
    node = cfg
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


def load_config_3d(path: Path) -> WaveConfig3D:
    """Parse a 3D YAML config into a WaveConfig3D."""
    if yaml is None:
        raise ImportError("PyYAML is required: pip install pyyaml")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    freq = _get(raw, "source", "frequency", default=1000.0)
    ssp_depths = tuple(_get(raw, "ssp", "depths", default=[0.0, 5.0]))
    ssp_speeds = tuple(_get(raw, "ssp", "speeds", default=[1500.0, 1500.0]))
    stencil_order = _get(raw, "grid", "stencil_order", default=2)

    grid = auto_resolve_grid_3d(
        frequency_hz=freq,
        c_min=min(ssp_speeds), c_max=max(ssp_speeds),
        width_m=_get(raw, "grid", "width_m"),
        height_m=_get(raw, "grid", "height_m"),
        depth_m=_get(raw, "grid", "depth_m"),
        dx=_get(raw, "grid", "dx", default=0),
        dy=_get(raw, "grid", "dy", default=0),
        dz=_get(raw, "grid", "dz", default=0),
        dt=_get(raw, "time", "dt", default=0),
        nx=_get(raw, "grid", "nx", default=0),
        ny=_get(raw, "grid", "ny", default=0),
        nz=_get(raw, "grid", "nz", default=0),
        cells_per_wavelength=_get(raw, "grid", "cells_per_wavelength", default=0),
        stencil_order=stencil_order,
    )

    duration = _get(raw, "time", "duration", default=0)
    steps_raw = _get(raw, "time", "steps", default=100000)
    if duration and duration > 0:
        steps = max(1, int(round(duration / grid["dt"])))
    else:
        steps = steps_raw

    # Source position: prefer physical coords (x/y/z in metres), fall back
    # to cell indices (ix/iy/iz), then default to grid centre.
    source_ix = _get(raw, "source", "ix")
    source_iy = _get(raw, "source", "iy")
    source_iz = _get(raw, "source", "iz")
    src_x = _get(raw, "source", "x")
    src_y = _get(raw, "source", "y")
    src_z = _get(raw, "source", "z")
    if src_x is not None:
        source_ix = int(round(float(src_x) / grid["dx"]))
    if src_y is not None:
        source_iy = int(round(float(src_y) / grid["dy"]))
    if src_z is not None:
        source_iz = int(round(float(src_z) / grid["dz"]))
    if source_ix is None:
        source_ix = grid["nx"] // 2
    if source_iy is None:
        source_iy = grid["ny"] // 2
    if source_iz is None:
        source_iz = grid["nz"] // 2
    source_ix = max(0, min(source_ix, grid["nx"] - 1))
    source_iy = max(0, min(source_iy, grid["ny"] - 1))
    source_iz = max(0, min(source_iz, grid["nz"] - 1))

    return WaveConfig3D(
        nx=grid["nx"], ny=grid["ny"], nz=grid["nz"],
        dx=grid["dx"], dy=grid["dy"], dz=grid["dz"],
        dt=grid["dt"], steps=steps,
        ssp_depths_m=ssp_depths, ssp_speeds_mps=ssp_speeds,
        wave_absorption_per_s=_get(raw, "model", "absorption", default=0.05),
        boundary_reflect_top=_get(raw, "boundary", "top", default=-0.98),
        boundary_reflect_bottom=_get(raw, "boundary", "bottom", default=0.99),
        boundary_reflect_left=_get(raw, "boundary", "left", default=0.99),
        boundary_reflect_right=_get(raw, "boundary", "right", default=0.99),
        boundary_reflect_front=_get(raw, "boundary", "front", default=0.0),
        boundary_reflect_back=_get(raw, "boundary", "back", default=0.0),
        stencil_order=stencil_order,
        backend=_get(raw, "backend", default="auto"),
        use_float32=_get(raw, "use_float32", default=True),
        source_ix=source_ix, source_iy=source_iy, source_iz=source_iz,
        source_amplitude_pa=_get(raw, "source", "amplitude", default=500.0),
        source_frequency_hz=freq,
        frame_stride=_get(raw, "output", "frame_stride", default=5),
    )


# ─── Replay loader ──────────────────────────────────────────────────────────


def _load_replay(replay_dir: Path):
    """Load saved 3D experiment data for playback."""
    npz_path = replay_dir / "frames.npz"
    bin_path = replay_dir / "_frames3d_cache.bin"
    meta_path = replay_dir / "_frames3d_cache_meta.json"

    if npz_path.exists():
        data = np.load(str(npz_path), allow_pickle=True)
        frames = list(data["frames"])
        meta = {k: data[k].item() if data[k].ndim == 0 else data[k]
                for k in data.files if k != "frames"}
        nz, ny, nx = frames[0].shape
        cfg = WaveConfig3D(
            nx=nx, ny=ny, nz=nz,
            dx=float(meta.get("dx", 0.05)),
            dy=float(meta.get("dy", 0.05)),
            dz=float(meta.get("dz", 0.05)),
            dt=float(meta.get("dt", 1e-4)),
            steps=1,
            source_ix=int(meta.get("source_ix", nx // 2)),
            source_iy=int(meta.get("source_iy", ny // 2)),
            source_iz=int(meta.get("source_iz", nz // 2)),
            source_frequency_hz=float(meta.get("source_frequency_hz", 1000)),
        )
        return cfg, frames, meta

    if bin_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            jmeta = json.load(f)
        nz = jmeta["nz"]; ny = jmeta["ny"]; nx = jmeta["nx"]
        count = jmeta["count"]
        dtype = np.dtype(jmeta.get("dtype", "float32"))
        reader = _DiskFrameReader3D(str(bin_path), count, nz, ny, nx, dtype)

        config_path = replay_dir / "config.yaml"
        if config_path.exists() and yaml is not None:
            cfg = load_config_3d(config_path)
            ds = jmeta.get("ds", 1)
            if ds > 1:
                cfg.nx = nx; cfg.ny = ny; cfg.nz = nz
                cfg.dx *= ds; cfg.dy *= ds; cfg.dz *= ds
        else:
            cfg = WaveConfig3D(nx=nx, ny=ny, nz=nz)

        meta = {
            "ds_factor": jmeta.get("ds", 1),
            "frame_times": np.arange(count) * cfg.dt * max(1, cfg.frame_stride),
        }
        return cfg, reader, meta

    raise FileNotFoundError(f"No replay data in {replay_dir}")


# ─── 3D GUI ─────────────────────────────────────────────────────────────────


COLORMAPS = ["coolwarm", "hot", "grays", "viridis"]
RENDER_METHODS = ["translucent", "mip", "iso"]

PROBE_COLORS_MPL = [
    "#ff4d4d", "#4dff4d", "#4d4dff",
    "#ffff4d", "#ff4dff", "#4dffff",
]


class ProbePlotWindow(QtWidgets.QWidget):
    """Separate Qt window showing probe pressure-vs-time curves."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Probes — Pressure vs Time")
        self.resize(600, 350)
        self.fig, self.ax = plt.subplots(figsize=(6, 3.2), tight_layout=True)
        self.canvas_mpl = FigureCanvas(self.fig)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas_mpl)
        self._lines: dict[int, matplotlib.lines.Line2D] = {}
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Pressure (Pa)")
        self.ax.set_title("Probe signals")
        self.ax.grid(True, alpha=0.3)

    def update_probes(self, probes: list[dict], dx: float, dy: float, dz: float):
        for i, pr in enumerate(probes):
            times = pr["times"]
            vals = pr["values"]
            if not times:
                continue
            color = PROBE_COLORS_MPL[i % len(PROBE_COLORS_MPL)]
            label = (f"P{i} ({pr['ix'] * dx:.2f}, "
                     f"{pr['iy'] * dy:.2f}, {pr['iz'] * dz:.2f})m")
            if i in self._lines:
                self._lines[i].set_data(times, vals)
            else:
                line, = self.ax.plot(times, vals, color=color, linewidth=1,
                                     label=label)
                self._lines[i] = line
                self.ax.legend(fontsize=7, loc="upper left")

        # Remove lines for deleted probes
        for idx in list(self._lines):
            if idx >= len(probes):
                self._lines[idx].remove()
                del self._lines[idx]
                self.ax.legend(fontsize=7, loc="upper left")

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas_mpl.draw_idle()

    def clear_all(self):
        for line in self._lines.values():
            line.remove()
        self._lines.clear()
        self.ax.legend(fontsize=7, loc="upper left")
        self.canvas_mpl.draw_idle()


class AcousticVolumeGUI:
    """Real-time 3D acoustic wave visualiser built on vispy."""

    def __init__(self, cfg: WaveConfig3D,
                 stepper: WaveStepper3D | None = None,
                 preloaded_frames=None,
                 replay_meta: dict | None = None):
        self.cfg = cfg
        self.stepper = stepper
        self.preloaded_frames = preloaded_frames
        self.replay_meta = replay_meta
        self.is_replay = preloaded_frames is not None
        self.total_replay_frames = len(preloaded_frames) if self.is_replay else 0

        self.paused = False
        self.spf = 1  # steps per frame
        self.frame_idx = 0
        self.cmap_idx = 0
        self.method_idx = 0
        self.vmax = 1.0
        self.vmax_manual = None  # None = auto-tracking, float = user-locked

        # Probe state
        self.probes: list[dict] = []
        self.probe_colors = [
            (1, 0.3, 0.3, 1), (0.3, 1, 0.3, 1), (0.3, 0.3, 1, 1),
            (1, 1, 0.3, 1), (1, 0.3, 1, 1), (0.3, 1, 1, 1),
        ]

        # Slice planes: {axis: (enabled, position_fraction)}
        self.slices = {"x": [False, 0.5], "y": [False, 0.5], "z": [False, 0.5]}
        self.active_slice_axis = "x"

        # FPS tracking
        self._fps_times: deque = deque(maxlen=30)
        self._last_time = _time.monotonic()
        self._plot_counter = 0

        self._probe_window: ProbePlotWindow | None = None

        self._build_scene()

    def _build_scene(self):
        cfg = self.cfg

        title = "3D Acoustic Wave"
        if self.is_replay:
            title += " (Replay)"
        if self.stepper:
            title += f" [{self.stepper.backend_name}]"

        self.canvas = scene.SceneCanvas(
            keys="interactive", title=title,
            size=(1200, 800), show=False)
        self.canvas.events.key_press.connect(self._on_key)
        self.canvas.events.mouse_press.connect(self._on_mouse_press)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(
            fov=45,
            distance=max(cfg.nx * cfg.dx, cfg.ny * cfg.dy, cfg.nz * cfg.dz) * 2.5,
            center=(cfg.nx * cfg.dx / 2, cfg.ny * cfg.dy / 2, cfg.nz * cfg.dz / 2),
        )

        # Initial volume data
        if self.is_replay and self.total_replay_frames > 0:
            vol = np.asarray(self.preloaded_frames[0], dtype=np.float32)
        else:
            vol = np.zeros((cfg.nz, cfg.ny, cfg.nx), dtype=np.float32)

        self.volume = visuals.Volume(
            vol, parent=self.view.scene,
            clim=(-1, 1), cmap=COLORMAPS[self.cmap_idx],
            method=RENDER_METHODS[self.method_idx],
            interpolation="linear",
        )
        # Scale to physical dimensions so the volume has correct aspect ratio
        self.volume.transform = scene.transforms.STTransform(
            scale=(cfg.dx, cfg.dy, cfg.dz))

        # Source marker — large, conspicuous, always on top
        self._src_world = np.array([
            cfg.source_ix * cfg.dx,
            cfg.source_iy * cfg.dy,
            cfg.source_iz * cfg.dz,
        ])
        self.source_marker = visuals.Markers(parent=self.view.scene)
        self.source_marker.set_data(
            self._src_world.reshape(1, 3),
            face_color=(1, 0.15, 0.15, 0.95), size=18,
            edge_color="white", edge_width=2.0,
            symbol="star",
        )
        self.source_marker.order = -1  # draw on top
        self.source_label = visuals.Text(
            f"SRC ({self._src_world[0]:.2f}, {self._src_world[1]:.2f}, "
            f"{self._src_world[2]:.2f})m",
            pos=self._src_world + np.array([0, 0, cfg.dz * 3]),
            font_size=8, color=(1, 0.4, 0.4, 1),
            anchor_x="center", anchor_y="bottom",
            parent=self.view.scene,
        )
        self.source_label.order = -1

        # Crosshair lines through source (subtle guide lines)
        w_phys = cfg.nx * cfg.dx
        h_phys = cfg.ny * cfg.dy
        d_phys = cfg.nz * cfg.dz
        sx, sy, sz = self._src_world
        cross_pts = np.array([
            [0, sy, sz], [w_phys, sy, sz],  # X-axis line
            [sx, 0, sz], [sx, h_phys, sz],  # Y-axis line
            [sx, sy, 0], [sx, sy, d_phys],  # Z-axis line
        ], dtype=np.float32)
        self.source_cross = visuals.Line(
            pos=cross_pts, color=(1, 0.3, 0.3, 0.25),
            parent=self.view.scene, method="gl", connect="segments")
        self.source_cross.order = -1

        # Bounding box
        w = w_phys
        h = h_phys
        d = d_phys
        box_verts = np.array([
            [0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0],
            [0, 0, d], [w, 0, d], [w, h, d], [0, h, d],
        ], dtype=np.float32)
        box_edges = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7],
        ], dtype=np.uint32)
        lines = box_verts[box_edges.ravel()].reshape(-1, 3)
        self.bbox = visuals.Line(pos=lines, color=(0.5, 0.5, 0.5, 0.6),
                                 parent=self.view.scene, method="gl",
                                 connect="segments")

        # Slice plane visuals (initially hidden)
        self._slice_visuals: dict[str, visuals.Plane | None] = {}
        self._build_slice_planes()

        # Probe markers + labels
        self.probe_markers = visuals.Markers(parent=self.view.scene)
        self.probe_markers.order = -1
        self.probe_labels: list[visuals.Text] = []

        # HUD text overlays
        self.hud_text = visuals.Text(
            "", pos=(10, 10), font_size=11,
            color="white", anchor_x="left", anchor_y="bottom",
            parent=self.canvas.scene)
        self.info_text = visuals.Text(
            "", pos=(10, 30), font_size=9,
            color=(0.8, 0.8, 0.8, 1), anchor_x="left", anchor_y="bottom",
            parent=self.canvas.scene)
        self.probe_text = visuals.Text(
            "", pos=(10, 50), font_size=9,
            color=(0.9, 0.9, 0.3, 1), anchor_x="left", anchor_y="bottom",
            parent=self.canvas.scene)

        # Axis labels
        for axis_label, pos, color in [
            ("X", (w + 0.3, 0, 0), (1, 0.3, 0.3, 1)),
            ("Y", (0, h + 0.3, 0), (0.3, 1, 0.3, 1)),
            ("Z", (0, 0, d + 0.3), (0.3, 0.3, 1, 1)),
        ]:
            visuals.Text(
                axis_label, pos=pos, font_size=10, color=color,
                parent=self.view.scene)

        # Timer for simulation updates
        self.timer = app.Timer(interval=1.0 / 30, connect=self._on_timer, start=False)

    def _build_slice_planes(self):
        cfg = self.cfg
        w = cfg.nx * cfg.dx
        h = cfg.ny * cfg.dy
        d = cfg.nz * cfg.dz

        # Remove existing slice visuals
        for key, vis in self._slice_visuals.items():
            if vis is not None:
                vis.parent = None
        self._slice_visuals.clear()

        for axis in ["x", "y", "z"]:
            enabled, frac = self.slices[axis]
            if enabled:
                if axis == "x":
                    pos = frac * w
                    plane_verts = np.array([
                        [pos, 0, 0], [pos, h, 0], [pos, h, d], [pos, 0, d]
                    ], dtype=np.float32)
                elif axis == "y":
                    pos = frac * h
                    plane_verts = np.array([
                        [0, pos, 0], [w, pos, 0], [w, pos, d], [0, pos, d]
                    ], dtype=np.float32)
                else:
                    pos = frac * d
                    plane_verts = np.array([
                        [0, 0, pos], [w, 0, pos], [w, h, pos], [0, h, pos]
                    ], dtype=np.float32)

                faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)
                colors = {"x": (1, 0.3, 0.3, 0.2),
                          "y": (0.3, 1, 0.3, 0.2),
                          "z": (0.3, 0.3, 1, 0.2)}
                mesh = visuals.Mesh(
                    vertices=plane_verts, faces=faces,
                    color=colors[axis],
                    parent=self.view.scene)
                self._slice_visuals[axis] = mesh
            else:
                self._slice_visuals[axis] = None

    def _on_timer(self, event):
        now = _time.monotonic()
        self._fps_times.append(now)

        if self.is_replay:
            if not self.paused:
                self.frame_idx = min(self.frame_idx + 1, self.total_replay_frames - 1)
            vol = np.asarray(self.preloaded_frames[self.frame_idx], dtype=np.float32)
        else:
            if not self.paused and self.stepper:
                self.stepper.step_n(self.spf)
            vol = self.stepper.pressure if self.stepper else np.zeros(
                (self.cfg.nz, self.cfg.ny, self.cfg.nx), dtype=np.float32)

        # Update volume with adaptive or manual color scaling
        absmax = float(np.abs(vol).max())
        if self.vmax_manual is not None:
            self.vmax = self.vmax_manual
        elif absmax > 1e-12:
            # Fast attack, moderate decay — keeps colors punchy
            if absmax > self.vmax:
                self.vmax = absmax
            else:
                self.vmax = self.vmax * 0.8 + absmax * 0.2
        self._current_vol = vol
        self.volume.set_data(vol)
        self.volume.clim = (-self.vmax, self.vmax)

        # Apply clipping for slice planes
        clipping_planes = []
        for axis, (enabled, frac) in self.slices.items():
            if enabled:
                if axis == "x":
                    pos = frac * self.cfg.nx * self.cfg.dx
                    clipping_planes.append(([pos, 0, 0], [-1, 0, 0]))
                elif axis == "y":
                    pos = frac * self.cfg.ny * self.cfg.dy
                    clipping_planes.append(([0, pos, 0], [0, -1, 0]))
                else:
                    pos = frac * self.cfg.nz * self.cfg.dz
                    clipping_planes.append(([0, 0, pos], [0, 0, -1]))

        # Update probe data
        for probe in self.probes:
            ix, iy, iz = probe["ix"], probe["iy"], probe["iz"]
            val = float(vol[iz, iy, ix])
            probe["values"].append(val)
            if self.is_replay:
                probe["times"].append(self.frame_idx)
            else:
                probe["times"].append(
                    self.stepper.current_time if self.stepper else 0)

        self._update_hud()

        # Update probe plot window every ~10 frames
        self._plot_counter += 1
        if self.probes and self._probe_window and self._plot_counter % 10 == 0:
            self._probe_window.update_probes(
                self.probes, self.cfg.dx, self.cfg.dy, self.cfg.dz)

        self.canvas.update()

    def _update_hud(self):
        # FPS calculation
        if len(self._fps_times) > 1:
            dt = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / max(dt, 1e-9)
        else:
            fps = 0

        if self.is_replay:
            t_str = f"Frame {self.frame_idx}/{self.total_replay_frames}"
            if self.replay_meta and "frame_times" in self.replay_meta:
                ft = self.replay_meta["frame_times"]
                if self.frame_idx < len(ft):
                    t_str += f"  t={float(ft[self.frame_idx]):.4f}s"
        else:
            step = self.stepper.current_step if self.stepper else 0
            t_sim = self.stepper.current_time if self.stepper else 0
            t_str = f"Step {step}  t={t_sim:.4f}s"

        status = "PAUSED" if self.paused else "RUNNING"
        self.hud_text.text = f"{t_str}  |  FPS: {fps:.0f}  |  {status}"

        method = RENDER_METHODS[self.method_idx]
        cmap = COLORMAPS[self.cmap_idx]
        spf_str = f"spf={self.spf}" if not self.is_replay else "replay"
        vmax_mode = "manual" if self.vmax_manual else "auto"
        vmax_str = f"vmax={self.vmax:.1e} ({vmax_mode})"
        slice_info = " ".join(
            f"{ax.upper()}={frac:.0%}"
            for ax, (en, frac) in self.slices.items() if en)
        self.info_text.text = (
            f"Method: {method}  Cmap: {cmap}  {spf_str}  {vmax_str}  "
            f"Slices: {slice_info or 'none'}  Active: {self.active_slice_axis.upper()}")

        if self.probes:
            lines = []
            for i, pr in enumerate(self.probes):
                pos_str = (f"({pr['ix'] * self.cfg.dx:.2f}, "
                           f"{pr['iy'] * self.cfg.dy:.2f}, "
                           f"{pr['iz'] * self.cfg.dz:.2f})m")
                val = pr["values"][-1] if pr["values"] else 0
                lines.append(f"P{i}: {pos_str} = {val:.1f} Pa")
            self.probe_text.text = "  |  ".join(lines)
        else:
            self.probe_text.text = "Click to place probe"

    def _on_key(self, event):
        key = event.key.name if hasattr(event.key, "name") else str(event.key)

        if key == "P":
            self.paused = not self.paused

        elif key in ("+", "="):
            self.spf = min(self.spf * 2, 256)

        elif key in ("-", "_"):
            self.spf = max(self.spf // 2, 1)

        elif key == "C":
            self.cmap_idx = (self.cmap_idx + 1) % len(COLORMAPS)
            self.volume.cmap = COLORMAPS[self.cmap_idx]

        elif key == "M":
            self.method_idx = (self.method_idx + 1) % len(RENDER_METHODS)
            self.volume.method = RENDER_METHODS[self.method_idx]

        elif key in ("X", "Y", "Z"):
            ax = key.lower()
            self.slices[ax][0] = not self.slices[ax][0]
            self.active_slice_axis = ax
            self._build_slice_planes()

        elif key in ("[", "]"):
            ax = self.active_slice_axis
            delta = 0.02 if key == "]" else -0.02
            self.slices[ax][1] = np.clip(self.slices[ax][1] + delta, 0.0, 1.0)
            self._build_slice_planes()

        elif key == "V":
            # Boost color intensity (halve vmax)
            self.vmax_manual = (self.vmax_manual or self.vmax) * 0.5
            self.vmax = self.vmax_manual

        elif key == "B":
            # Dim color intensity (double vmax)
            self.vmax_manual = (self.vmax_manual or self.vmax) * 2.0
            self.vmax = self.vmax_manual

        elif key == "A":
            # Toggle auto color scaling
            self.vmax_manual = None

        elif key == "R":
            cfg = self.cfg
            self.view.camera.center = (
                cfg.nx * cfg.dx / 2,
                cfg.ny * cfg.dy / 2,
                cfg.nz * cfg.dz / 2)
            self.view.camera.distance = (
                max(cfg.nx * cfg.dx, cfg.ny * cfg.dy, cfg.nz * cfg.dz) * 2.5)

        elif key == "Backspace":
            if self.probes:
                self.probes.pop()
                self._update_probe_markers()
                if self._probe_window:
                    if self.probes:
                        self._probe_window.update_probes(
                            self.probes, self.cfg.dx, self.cfg.dy, self.cfg.dz)
                    else:
                        self._probe_window.clear_all()

        elif key in ("Q", "Escape"):
            self.timer.stop()
            app.quit()

    def _on_mouse_press(self, event):
        if event.button != 1:
            return
        if not event.modifiers or "Shift" not in [m.name for m in event.modifiers]:
            return

        cfg = self.cfg
        try:
            # Get the transform from canvas pixels → scene (world) coordinates
            tr = self.canvas.scene.node_transform(self.view.scene)
            # Map click position at near plane (z=0) and far plane (z=1)
            p0 = tr.map((*event.pos, 0, 1))[:3]
            p1 = tr.map((*event.pos, 1, 1))[:3]
            ray_dir = p1 - p0
            ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)

            # Intersect ray with the volume's center-z plane
            z_mid = cfg.nz * cfg.dz * 0.5
            if abs(ray_dir[2]) > 1e-9:
                t = (z_mid - p0[2]) / ray_dir[2]
            else:
                t = 0
            hit = p0 + ray_dir * t

            ix = int(np.clip(hit[0] / cfg.dx, 0, cfg.nx - 1))
            iy = int(np.clip(hit[1] / cfg.dy, 0, cfg.ny - 1))
            iz = int(np.clip(hit[2] / cfg.dz, 0, cfg.nz - 1))
        except Exception:
            # Fallback: place at volume centre
            ix, iy, iz = cfg.nx // 2, cfg.ny // 2, cfg.nz // 2

        self._add_probe(ix, iy, iz)

    def _add_probe(self, ix: int, iy: int, iz: int):
        color = self.probe_colors[len(self.probes) % len(self.probe_colors)]
        self.probes.append({
            "ix": ix, "iy": iy, "iz": iz,
            "color": color,
            "values": [],
            "times": [],
        })
        self._update_probe_markers()

        # Open probe plot window on first probe
        if self._probe_window is None:
            self._probe_window = ProbePlotWindow()
        self._probe_window.show()
        self._probe_window.raise_()

        cfg = self.cfg
        print(f"Probe {len(self.probes) - 1} at cell ({ix},{iy},{iz}) = "
              f"({ix * cfg.dx:.2f}, {iy * cfg.dy:.2f}, "
              f"{iz * cfg.dz:.2f})m")

    def _update_probe_markers(self):
        # Remove stale labels
        while len(self.probe_labels) > len(self.probes):
            lbl = self.probe_labels.pop()
            lbl.parent = None

        if not self.probes:
            self.probe_markers.set_data(np.zeros((0, 3)))
            return

        cfg = self.cfg
        positions = np.array([
            [p["ix"] * cfg.dx, p["iy"] * cfg.dy, p["iz"] * cfg.dz]
            for p in self.probes
        ], dtype=np.float32)
        colors = np.array([p["color"] for p in self.probes], dtype=np.float32)
        self.probe_markers.set_data(
            positions, face_color=colors, size=16,
            edge_color="white", edge_width=2.0,
            symbol="diamond",
        )

        # Create / update text labels above each probe
        for i, pr in enumerate(self.probes):
            pos = np.array([
                pr["ix"] * cfg.dx,
                pr["iy"] * cfg.dy,
                pr["iz"] * cfg.dz + cfg.dz * 3,
            ])
            txt = (f"P{i} ({pr['ix'] * cfg.dx:.2f}, "
                   f"{pr['iy'] * cfg.dy:.2f}, {pr['iz'] * cfg.dz:.2f})m")
            if i < len(self.probe_labels):
                self.probe_labels[i].text = txt
                self.probe_labels[i].pos = pos
            else:
                clr = PROBE_COLORS_MPL[i % len(PROBE_COLORS_MPL)]
                lbl = visuals.Text(
                    txt, pos=pos, font_size=7, color=clr,
                    anchor_x="center", anchor_y="bottom",
                    parent=self.view.scene,
                )
                lbl.order = -1
                self.probe_labels.append(lbl)

    def run(self):
        self.canvas.show()
        self.timer.start()
        app.run()


# ─── CLI ────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="3D acoustic wave GUI (vispy)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", type=str, help="Path to 3D YAML config")
    g.add_argument("--replay", type=str, help="Experiment dir to replay")
    p.add_argument("--spf", type=int, default=1, help="Initial steps per frame")
    return p.parse_args()


def main():
    args = parse_args()

    if args.replay:
        replay_dir = Path(args.replay)
        if not replay_dir.is_absolute():
            replay_dir = PROJECT_ROOT / replay_dir
        print(f"Loading replay: {replay_dir}")
        cfg, frames, meta = _load_replay(replay_dir)
        print(f"Loaded {len(frames)} frames, grid {cfg.nx}x{cfg.ny}x{cfg.nz}")
        gui = AcousticVolumeGUI(cfg, preloaded_frames=frames, replay_meta=meta)
    else:
        config_path = Path(args.config)
        if not config_path.is_absolute():
            config_path = PROJECT_ROOT / config_path
        print(f"Loading config: {config_path}")
        cfg = load_config_3d(config_path)
        print(f"Grid: {cfg.nx}x{cfg.ny}x{cfg.nz} = "
              f"{cfg.nx * cfg.ny * cfg.nz / 1e6:.2f}M cells")
        print(f"Cell size: dx={cfg.dx:.4f} dy={cfg.dy:.4f} dz={cfg.dz:.4f}")
        print(f"dt={cfg.dt:.6e}s, source={cfg.source_frequency_hz:.0f}Hz")

        stepper = WaveStepper3D(cfg)
        print(f"Backend: {stepper.backend_name}")

        gui = AcousticVolumeGUI(cfg, stepper=stepper)
        gui.spf = args.spf

    gui.run()


if __name__ == "__main__":
    main()
