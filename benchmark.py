"""
EchoFlow full-pipeline benchmark.

Runs all three stages (raw -> preprocessing -> inference) inside Docker,
varying MAX_WORKERS to measure throughput and scaling.

Prerequisites:
    - Docker and docker compose installed
    - Benchmark data downloaded: bash download_bench_data.sh
    - Docker images built: docker compose build

Usage:
    python benchmark.py [--data-dir data/benchmark/input] [--workers 1,2,4,8]
"""

import os
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

COMPOSE_FILES = ["docker-compose.yml", "docker-compose.benchmark.yml"]


def get_machine_info():
    """Gather machine specs for the report header."""
    lines = [
        f"- **OS:** {platform.system()} {platform.release()}",
        f"- **CPU:** {platform.processor() or platform.machine()}",
        f"- **Cores:** {os.cpu_count()}",
    ]
    try:
        gpu = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                timeout=5,
            )
            .strip()
            .split("\n")[0]
        )
        lines.append(f"- **GPU:** {gpu}")
    except (FileNotFoundError, subprocess.SubprocessError):
        lines.append("- **GPU:** None (CPU only)")
    return "\n".join(lines)


def count_files(directory, pattern):
    """Count files matching glob pattern in directory."""
    return len(list(Path(directory).rglob(pattern)))


def run_stage(service, env_overrides):
    """Run a docker compose service with -e flags and return elapsed seconds."""
    cmd = ["docker", "compose"]
    for f in COMPOSE_FILES:
        cmd.extend(["-f", f])
    cmd.extend(["run", "--rm"])
    for k, v in env_overrides.items():
        cmd.extend(["-e", f"{k}={v}"])
    cmd.append(service)

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    if result.returncode != 0:
        print(f"  WARNING: {service} exited with code {result.returncode}")
        if result.stderr:
            print(result.stderr[-500:])

    return elapsed


def clean_dir(path, protected):
    """Remove and recreate a directory. Docker output is root-owned."""
    path = Path(path).resolve()
    assert path != Path(protected).resolve(), f"Refusing to delete input dir: {path}"
    if path.exists():
        # Docker creates files as root; use a container to wipe contents
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "-v",
                f"{path}:/clean",
                "alpine",
                "sh",
                "-c",
                "rm -rf /clean/*",
            ],
            capture_output=True,
        )
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class StageResult:
    workers: int
    t1: float
    t2: float
    t3: float
    total: float
    per_file: float
    n_raw: int


def run_benchmark(
    data_dir, worker_counts, batch_size=4, device="cuda", downsample_size=1000
):
    """Run full pipeline for each worker count and collect timings."""
    data_dir = Path(data_dir).resolve()
    n_raw = count_files(data_dir, "*.raw")
    raw_size_gb = sum(f.stat().st_size for f in data_dir.rglob("*.raw")) / 1e9

    if n_raw == 0:
        print("ERROR: No .raw files found. Run: bash download_bench_data.sh")
        return

    print("# EchoFlow Benchmark\n")
    print("## Machine\n")
    print(get_machine_info())
    print("\n## Dataset\n")
    print(f"- **Files:** {n_raw} EK80 `.raw` echograms")
    print(f"- **Total size:** {raw_size_gb:.1f} GB")
    print("- **Source:** NOAA WCSD public bucket (SH2306 cruise)\n")

    bench_root = data_dir.parent  # data/benchmark/
    raw_out = bench_root / "raw_consumer"
    pre_out = bench_root / "preprocessing"
    inf_out = bench_root / "inference"
    log_out = bench_root / "log"

    results = []

    for workers in worker_counts:
        print(f"\n### MAX_WORKERS={workers}\n")

        clean_dir(raw_out, protected=data_dir)
        clean_dir(pre_out, protected=data_dir)
        clean_dir(inf_out, protected=data_dir)
        clean_dir(log_out, protected=data_dir)

        env = {
            "MAX_WORKERS": str(workers),
            "BATCH_SIZE": str(batch_size),
            "KEEP_INTERMEDIATES": "false",
            "DEVICE": device,
            "DOWNSAMPLE_SIZE": str(downsample_size),
        }

        # Stage 1: raw -> .nc
        print("  Stage 1 (raw -> netCDF) ...", end=" ", flush=True)
        t1 = run_stage("raw", env)
        n_nc = count_files(raw_out, "*.nc")
        print(f"{t1:.1f}s ({n_nc} files)")

        # Stage 2: .nc -> .png
        print("  Stage 2 (preprocessing)  ...", end=" ", flush=True)
        t2 = run_stage("preprocessing", env)
        n_png = count_files(pre_out, "*.png")
        print(f"{t2:.1f}s ({n_png} files)")

        # Stage 3: .png -> attention maps
        print("  Stage 3 (inference)      ...", end=" ", flush=True)
        t3 = run_stage("infer", env)
        n_attn = count_files(inf_out, "*.png")
        print(f"{t3:.1f}s ({n_attn} files)")

        total = t1 + t2 + t3
        per_file = total / n_raw if n_raw else 0

        results.append(
            StageResult(
                workers=workers,
                t1=t1,
                t2=t2,
                t3=t3,
                total=total,
                per_file=per_file,
                n_raw=n_raw,
            )
        )

    # Summary table
    print("\n## Results\n")
    print(
        "| Workers | Stage 1 (s) | Stage 2 (s) | Stage 3 (s) | Total (s) | s/file | Speedup |"
    )
    print(
        "|---------|-------------|-------------|-------------|-----------|--------|---------|"
    )
    baseline = results[0].total if results else 1
    for r in results:
        speedup = baseline / r.total if r.total > 0 else 0
        print(
            f"| {r.workers} | {r.t1:.1f} | {r.t2:.1f} | {r.t3:.1f} "
            f"| {r.total:.1f} | {r.per_file:.2f} | {speedup:.2f}x |"
        )

    # Extrapolation
    if results:
        best = min(results, key=lambda r: r.per_file)
        secs_per_file = best.per_file
        avg_file_gb = raw_size_gb / n_raw
        files_per_tb = 1000 / avg_file_gb if avg_file_gb > 0 else 0
        hours_per_tb = (secs_per_file * files_per_tb) / 3600

        print("\n## Extrapolation\n")
        print(
            f"- **Best throughput:** {secs_per_file:.2f} s/file at {best.workers} workers"
        )
        print(f"- **Average file size:** {avg_file_gb * 1000:.0f} MB")
        print(f"- **Estimated time for 1 TB:** {hours_per_tb:.1f} hours")
        print(
            f"- **Estimated time for 10 TB:** {hours_per_tb * 10:.0f} hours "
            f"({hours_per_tb * 10 / 24:.1f} days)"
        )

    print("\n*Run `python benchmark.py` to regenerate.*")


if __name__ == "__main__":
    import tyro

    @dataclass
    class Args:
        """EchoFlow full-pipeline benchmark."""

        data_dir: str = "data/benchmark/input"
        """Directory containing .raw files."""
        workers: str = "1,2,4,8"
        """Comma-separated worker counts to test."""
        batch_size: int = 4
        """Inference batch size."""
        device: str = "cuda"
        """Inference device (cuda or cpu)."""
        downsample_size: int = 1000
        """Image size for inference (pixels)."""

    args = tyro.cli(Args)
    worker_counts = [int(w) for w in args.workers.split(",")]
    run_benchmark(
        args.data_dir, worker_counts, args.batch_size, args.device, args.downsample_size
    )
