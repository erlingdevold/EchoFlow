# Usage

## Running the full pipeline

```bash
docker compose up --build raw preprocessing infer
```

This runs all three stages sequentially. Each stage reads from and writes to bind-mounted directories under `./data/`.

## Running individual stages

| Stage | Command |
|-------|---------|
| Stage 1 — Conversion | `docker compose up --build raw` |
| Stage 2 — Pre-processing | `docker compose up --build preprocessing` |
| Stage 3 — Inference | `docker compose up --build infer` |

## Environment variables

### Global

| Variable | Default | Description |
|----------|---------|-------------|
| `MAX_WORKERS` | CPU count | Max parallel workers for stages 1–2 |

### Stage 2: Preprocessing

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/data/processed` | Input directory for .nc files |
| `OUTPUT_DIR` | `/data/test_imgs` | Output directory for PNGs |
| `LOG_DIR` | `.` | Log directory |
| `KEEP_INTERMEDIATES` | `true` | Preserve .nc files after processing |

### Stage 3: Inference

| Variable | Default | Description |
|----------|---------|-------------|
| `INPUT_DIR` | `/data/test_imgs` | Input directory for PNGs |
| `OUTPUT_DIR` | `/data/inference` | Output directory for attention maps |
| `LOG_DIR` | `.` | Log directory |
| `ARCH` | `vit_small` | Vision Transformer architecture |
| `PATCH_SZ` | `8` | Patch size for ViT |
| `DOWNSAMPLE_SIZE` | `5000` | Image downsample dimension |
| `BATCH_SIZE` | `4` | Images per GPU forward pass |
| `DEVICE` | auto | `cuda`, `cpu`, or auto-detect |

## GPU support

To enable GPU passthrough in Docker, uncomment the `deploy` section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - capabilities: [gpu]
```

## Monitor dashboard

Once the pipeline is running, a progress monitor is available at **http://localhost:8050**. It tracks file counts and tail-logs for each stage.

## Output structure

```
data/
├── raw_consumer/    # Stage 1: converted NetCDF files
├── preprocessing/   # Stage 2: echogram PNGs
└── inference/       # Stage 3: attention map PNGs
```

Each run creates a subdirectory named after the input file.
