# %%
"""
Benchmark script for EchoFlow pipeline stages.

Generates a markdown-formatted throughput table for the README.
Run with: python benchmark.py

Requires a sample .nc file in data/raw_consumer/ and corresponding
PNG output in data/preprocessing/ for stages 2 and 3.
"""

import os
import platform
import time
from pathlib import Path

import numpy as np
import xarray as xr


def get_machine_info():
    """Print machine specs."""
    import torch

    lines = [
        f"- **OS:** {platform.system()} {platform.release()}",
        f"- **CPU:** {platform.processor() or platform.machine()}",
        f"- **Cores:** {os.cpu_count()}",
        f"- **GPU:** {'CUDA ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None (CPU only)'}",
    ]
    return "\n".join(lines)


def create_synthetic_nc(tmp_dir, n_files=1):
    """Create synthetic .nc files for benchmarking."""
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    files = []
    rng = np.random.default_rng(42)

    for i in range(n_files):
        freqs = [38000.0, 70000.0]
        n_pings = 200
        n_depths = 500
        ping_time = np.array(
            [np.datetime64("2024-01-01") + np.timedelta64(j, "s") for j in range(n_pings)]
        )
        depth = np.linspace(0, 500, n_depths)
        sv_linear = 10 ** (rng.uniform(-80, -30, size=(2, n_pings, n_depths)) / 10)
        sv = xr.DataArray(sv_linear, coords=[freqs, ping_time, depth], dims=["frequency", "ping_time", "depth"])
        ds = xr.Dataset({"Sv": sv})
        path = tmp_dir / f"bench_{i:04d}.nc"
        ds.to_netcdf(path)
        files.append(path)

    return files


def bench_preprocessing(nc_files, output_dir, workers):
    """Benchmark stage 2 preprocessing with given worker count."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "preprocessing"))
    os.environ["OUTPUT_DIR"] = str(output_dir)
    os.environ["LOG_DIR"] = str(output_dir)

    import preprocessing as pp

    start = time.perf_counter()
    for f in nc_files:
        pp.sv_to_jpg(f, estimate_bot=True)
    elapsed = time.perf_counter() - start
    return elapsed


def bench_inference(png_dir, output_dir, batch_size):
    """Benchmark stage 3 inference with given batch size."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent / "inference"))

    import inspect_attention as ia

    device, model = ia.setup_device_and_model(
        arch="vit_tiny", patch_size=16, pretrained_weights="checkpoint.pth"
    )

    all_pngs = list(Path(png_dir).rglob("*.png"))
    if not all_pngs:
        return 0.0

    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_size = (256, 256)
    start = time.perf_counter()
    for i in range(0, len(all_pngs), batch_size):
        batch = all_pngs[i:i + batch_size]
        tensors = torch.stack([ia.process_image(f, image_size) for f in batch]).to(ia.DEVICE)
        with torch.no_grad():
            ia.get_attention_maps(model, tensors, image_size=image_size, patch_size=16)
    elapsed = time.perf_counter() - start
    return elapsed


if __name__ == "__main__":
    import tempfile

    print("# EchoFlow Benchmark\n")
    print("## Machine\n")
    print(get_machine_info())
    print()

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        nc_dir = tmp / "nc"
        pre_out = tmp / "pre_out"
        inf_out = tmp / "inf_out"
        pre_out.mkdir()
        inf_out.mkdir()

        print("Creating synthetic data...")
        nc_files = create_synthetic_nc(nc_dir, n_files=4)

        print("\n## Stage 2: Preprocessing\n")
        print("| Files | Time (s) |")
        print("|-------|----------|")
        t = bench_preprocessing(nc_files, pre_out, workers=1)
        print(f"| {len(nc_files)} | {t:.2f} |")

        print("\n## Stage 3: Inference\n")
        print("| Batch size | Files | Time (s) |")
        print("|------------|-------|----------|")
        for bs in [1, 2, 4]:
            t = bench_inference(pre_out, inf_out, batch_size=bs)
            pngs = list(pre_out.rglob("*.png"))
            print(f"| {bs} | {len(pngs)} | {t:.2f} |")

    print("\n*Run `python benchmark.py` to regenerate.*")
