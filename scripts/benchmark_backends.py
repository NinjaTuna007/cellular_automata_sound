from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from pressure_transfer_ca import PressureCASimConfig, run_pressure_transfer_ca


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark pressure-transfer CA backends.")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--nx", type=int, default=220)
    parser.add_argument("--ny", type=int, default=140)
    parser.add_argument("--dx", type=float, default=2.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--model", type=str, default="wave", choices=["wave", "transfer"])
    parser.add_argument("--freq-hz", type=float, default=24_000.0)
    parser.add_argument("--skip-gpu", action="store_true")
    return parser.parse_args()


def run_one(backend: str, base_cfg: PressureCASimConfig) -> float:
    cfg = PressureCASimConfig(**{**base_cfg.__dict__, "backend": backend, "frame_stride": 10_000_000})
    t0 = time.perf_counter()
    out = run_pressure_transfer_ca(cfg)
    elapsed = time.perf_counter() - t0
    print(f"{backend:>4} => used={out['backend_used']:<20} elapsed={elapsed:.3f}s")
    return elapsed


def main():
    args = parse_args()
    ssp_speeds = (1535.0, 1518.0, 1492.0, 1503.0, 1520.0)
    cmax = max(ssp_speeds)
    dt_auto = 0.45 / (cmax * ((1.0 / (args.dx * args.dx) + 1.0 / (args.dy * args.dy)) ** 0.5))
    base_cfg = PressureCASimConfig(
        nx=args.nx,
        ny=args.ny,
        dx=args.dx,
        dy=args.dy,
        dt=dt_auto,
        steps=args.steps,
        propagation_model=args.model,
        transfer_fraction=0.42,
        damping=0.004,
        overpressure_only=False,
        use_impedance_interface=True,
        source_frequency_hz=args.freq_hz,
        ssp_speeds_mps=ssp_speeds,
    )

    cpu_elapsed = run_one("cpu", base_cfg)
    if not args.skip_gpu:
        try:
            gpu_elapsed = run_one("gpu", base_cfg)
            if gpu_elapsed > 0:
                print(f"Speedup (CPU/GPU): {cpu_elapsed / gpu_elapsed:.2f}x")
        except Exception as exc:
            print(f"GPU benchmark skipped due to runtime error: {exc}")


if __name__ == "__main__":
    main()

