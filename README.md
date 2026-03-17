# Echoflow
[![CI](https://github.com/erlingdevold/EchoFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/erlingdevold/EchoFlow/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15634054.svg)](https://doi.org/10.5281/zenodo.15634054)
[![status](https://joss.theoj.org/papers/5c9046c3818c08881b51acf6be8d79dc/status.svg)](https://joss.theoj.org/papers/5c9046c3818c08881b51acf6be8d79dc)

> Do you have Terabytes of unprocessed Sonar data? Use this for easy viewing and inference!

EchoFlow is a three-stage containerised pipeline that converts raw Kongsberg EK80 echosounder files into echograms and DINO ViT attention maps.

**Stages:**
1. **Conversion** (`raw`) — decodes `.raw` pings to volume-backscattering strength NetCDF files via pyEcholab.
2. **Pre-processing** (`preprocessing`) — contrast-stretches and tiles echograms as PNGs.
3. **Inference** (`infer`) — runs a DINO Vision Transformer to produce per-patch attention heat-maps.

## Features

- **Preprocessing**: Prepares the data for inference.
- **Inference**: Utilises DINO for visual inference and attention inspection.
- **Docker Support**: Run the pipeline in an isolated Docker environment for consistency and ease of use.

## Repository Structure

```
.
├── raw_consumer/                # process raw to xarray
│   ├── Dockerfile.raw            # Dockerfile for volumetric backscatter computing with pyEcholab
│   ├── preprocessing.py          # Script for converting raw file to volumetric backscatter cubes
├── preprocessing/               # Preprocessing components
│   ├── Dockerfile.preprocessing  # Dockerfile for preprocessing
│   ├── preprocessing.py          # Script for preprocessing input data
├── inference/                   # Inference components
│   ├── Dockerfile.infer          # Dockerfile for inference
│   ├── attention_inspect.py      # Script for inspecting attention maps
│   ├── inspect_attention.py      # Main script for running inference and inspection
│   ├── requirements.txt          # Python dependencies for inference demo
│   ├── utils.py                  # Utility functions for inference demo
│   ├── vision_transformer.py     # DINO Vision Transformer model implementation
├── docker-compose.yml            # Docker Compose file to run the entire pipeline
├── entrypoint.sh                 # Entrypoint script for Docker container
├── infer.py                      # Main script to run inference outside Docker
├── run_docker.sh                 # Script to run the pipeline using Docker
├── watchdog.py                   # Script to watch for changes in the pipeline
```

## Installation

### Prerequisites

- Docker ≥ 24
- Docker Compose v2 (`docker compose` — note: no hyphen)
- Git
- AWS CLI (for downloading the sample input file)

### Clone with submodules

```bash
git clone --recurse-submodules https://github.com/erlingdevold/EchoFlow.git
```

If you have already cloned without submodules:

```bash
git submodule update --init --recursive
```

### Without Docker (native install)

The raw_consumer stage requires HDF5 and netCDF C libraries:

- **macOS:** `brew install hdf5 netcdf`
- **Ubuntu/Debian:** `apt-get install libhdf5-dev libnetcdf-dev`
- **Docker images include these already** — no action needed for containerised use.

Then install Python dependencies for the stages you need:

```bash
pip install -r raw_consumer/requirements.txt
pip install -r preprocessing/requirements.txt
pip install -r inference/requirements.txt
```

## Populate input

The command below fetches a publicly available NOAA EK80 test file (~105 MB) into `data/input/` and initialises git submodules before building the Docker images.

```bash
aws s3 cp --no-sign-request \
  "s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw" \
  data/input/

touch ./inference/checkpoint.pth

git submodule sync --recursive
```

## Setup

### Running with Docker

#### Run the full pipeline

```bash
docker compose up --build
```

This builds and starts all three stages (Conversion, Pre-processing, Inference) plus the progress monitor.

#### Run individual stages

You can also run each stage independently:

| Stage | Command |
|-------|---------|
| Stage 1 — Conversion | `docker compose up --build raw` |
| Stage 2 — Pre-processing | `docker compose up --build preprocessing` |
| Stage 3 — Inference | `docker compose up --build infer` |
| All stages (no monitor) | `docker compose up --build raw preprocessing infer` |

Each stage reads from and writes to bind-mounted directories under `./data/`, so stages can be run in sequence without rebuilding upstream containers.

### Monitor dashboard

Once the pipeline is running, a progress monitor is available at **http://localhost:8050**.

The dashboard is a **pipeline progress monitor** — it tracks file counts and tail-logs for each stage. It is not an inference viewer. Actual outputs are written to:

- `data/raw_consumer/` — converted NetCDF files (Stage 1 output)
- `data/preprocessing/` — echogram PNGs (Stage 2 output)
- `data/inference/` — attention map PNGs (Stage 3 output)

## Performance / parallelism

Stages 1 (conversion) and 2 (pre-processing) use process pools to parallelise work across files — controlled by the `MAX_WORKERS` env var (defaults to CPU count). Stage 3 (inference) uses batched GPU inference with a configurable `BATCH_SIZE` (default 4) and auto-detects CUDA when available. Because `.raw` files are large XML datagrams, file I/O is the primary bottleneck for stages 1–2; adding CPU cores within a node yields proportional throughput gains. Parallelism is intra-node only and there is no built-in cluster or scheduler integration.

Run `python benchmark.py` to measure throughput on your hardware. Example output:

<!-- Run `python benchmark.py` to regenerate this table -->
| Stage | Metric | Value |
|-------|--------|-------|
| Preprocessing (4 files) | Time | *run benchmark* |
| Inference batch=1 | Time | *run benchmark* |
| Inference batch=4 | Time | *run benchmark* |

## ENV variables

### Global
- `MAX_WORKERS` — max parallel workers for stages 1–2 (default: CPU count)

### `watchdog.py`
- `LOG_DIR` (default: `"/data/log"`)
- `INPUT_DIR` (default: `"/data/sonar"`)
- `OUTPUT_DIR` (default: `"/data/processed"`)

### `raw.py` (Stage 1)
- `INPUT_DIR` (default: `"/data/sonar"`)
- `OUTPUT_DIR` (default: `"/data/processed"`)
- `LOG_DIR` (default: `"log"`)

### `preprocessing.py` (Stage 2)
- `INPUT_DIR` (default: `"/data/processed"`)
- `OUTPUT_DIR` (default: `"/data/test_imgs"`)
- `LOG_DIR` (default: `"."`)
- `KEEP_INTERMEDIATES` — preserve `.nc` files after processing (default: `"true"`)

### `inspect_attention.py` (Stage 3)
- `INPUT_DIR` (default: `"/data/test_imgs"`)
- `OUTPUT_DIR` (default: `"/data/inference"`)
- `LOG_DIR` (default: `"."`)
- `PATCH_SZ` (default: `8`)
- `ARCH` (default: `'vit_small'`)
- `DOWNSAMPLE_SIZE` (default: `5000`)
- `BATCH_SIZE` — images per GPU forward pass (default: `4`)
- `DEVICE` — `"cuda"`, `"cpu"`, or auto-detect (default: auto)

## Output

The output of the inference step, including generated attention maps and transformed images, will be saved in `data/inference/`. Each run creates a subdirectory named after the input file for organised output management.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on reporting bugs, suggesting features, and submitting pull requests.

## License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Acknowledgements

This pipeline uses the DINO Vision Transformer for attention-based image analysis. The implementation is based on research from the original DINO paper by Facebook AI Research (FAIR).
