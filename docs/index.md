# EchoFlow

> Do you have terabytes of unprocessed sonar data? Use this for easy viewing and inference!

**EchoFlow** is a three-stage, containerised pipeline that converts raw Kongsberg EK80 echosounder files into echograms and DINO ViT attention maps.

## Stages

1. **Conversion** (`raw`) — decodes `.raw` pings to volume-backscattering strength NetCDF files via PyEcholab.
2. **Pre-processing** (`preprocessing`) — contrast-stretches and tiles echograms as PNGs.
3. **Inference** (`infer`) — runs a DINO Vision Transformer to produce per-patch attention heat-maps.

## Features

- Fully containerised with Docker Compose
- Configurable parallelism (`MAX_WORKERS`) and batched GPU inference (`BATCH_SIZE`)
- Watchdog-triggered processing when new `.raw` files arrive
- Progress monitoring dashboard at `http://localhost:8050`
- CI pipeline that validates end-to-end on a public NOAA test file

## Quick start

```bash
git clone --recurse-submodules https://github.com/erlingdevold/EchoFlow.git
cd EchoFlow

# Fetch a sample file
aws s3 cp --no-sign-request \
  "s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw" \
  data/input/
touch inference/checkpoint.pth

# Run the pipeline
docker compose up --build raw preprocessing infer
```

Outputs land in `data/inference/`.
