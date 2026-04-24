#!/usr/bin/env python3
"""Run multiple experiment configs in parallel across CPU cores.

Usage:
  python scripts/sweep.py configs/a.yaml configs/b.yaml configs/c.yaml
  python scripts/sweep.py configs/*.yaml
  python scripts/sweep.py configs/*.yaml --workers 8
  python scripts/sweep.py configs/*.yaml --workers 8 --no-gif

Each config is run as an independent experiment via run_experiment.py.
Worker count defaults to (CPU cores - 2) so you still have headroom.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    p = argparse.ArgumentParser(
        description="Run multiple experiment configs in parallel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("configs", nargs="+", help="Paths to YAML config files")
    p.add_argument("--workers", type=int, default=0,
                    help="Max parallel workers. 0 → (CPU cores - 2)")
    p.add_argument("--no-gif", action="store_true", help="Pass --no-gif to each run")
    p.add_argument("--backend", type=str, default=None, choices=["auto", "cpu", "gpu"],
                    help="Force a specific backend for all runs")
    return p.parse_args()


def run_single(config_path: str, extra_args: list) -> dict:
    """Run one experiment in a subprocess, return summary."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_experiment.py"),
        "--config", config_path,
        *extra_args,
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
    elapsed = time.perf_counter() - t0
    return {
        "config": config_path,
        "elapsed": elapsed,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    args = parse_args()

    ncpu = os.cpu_count() or 4
    workers = args.workers if args.workers > 0 else max(1, ncpu - 2)
    configs = args.configs

    extra = []
    if args.no_gif:
        extra.append("--no-gif")
    if args.backend:
        extra.extend(["--backend", args.backend])

    print(f"Sweep: {len(configs)} configs, {workers} parallel workers, {ncpu} CPU threads available")
    print("─" * 70)

    t_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_single, cfg, extra): cfg for cfg in configs}
        for future in as_completed(futures):
            result = future.result()
            cfg_name = Path(result["config"]).name
            status = "OK" if result["returncode"] == 0 else "FAIL"
            print(f"  [{status}] {cfg_name:40s} {result['elapsed']:7.1f}s")
            if result["returncode"] != 0:
                for line in result["stderr"].strip().split("\n")[-5:]:
                    print(f"        {line}")

    total = time.perf_counter() - t_start
    print("─" * 70)
    print(f"Total wall time: {total:.1f}s for {len(configs)} experiments "
          f"({total / len(configs):.1f}s avg, {workers} workers)")


if __name__ == "__main__":
    main()
