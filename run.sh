#!/bin/bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"

# Stage 1: raw conversion
#container run --rm -m 8g \
#    -v "$ROOT/data/input:/app/input" \
#    -v "$ROOT/data/raw_consumer:/app/output" \
#    -v "$ROOT/raw_consumer/log:/app/log" \
#    -e INPUT_DIR=/app/input \
#    -e OUTPUT_DIR=/app/output \
#    -e LOG_DIR=/app/log \
#    -e PIPELINE_STAGE=RAW \
#    echoflow-raw python main.py

# Stage 2: preprocessing
#container run --rm -m 4g \
#    -v "$ROOT/data/raw_consumer:/app/input" \
#    -v "$ROOT/data/preprocessing:/app/output" \
#    -v "$ROOT/preprocessing/log:/app/log" \
#    -e INPUT_DIR=/app/input \
#    -e OUTPUT_DIR=/app/output \
#    -e LOG_DIR=/app/log \
#    -e PIPELINE_STAGE=PRE \
#    echoflow-pre python main.py

# Stage 3: inference
container run --rm -m 8g \
    -v "$ROOT/data/preprocessing:/app/input" \
    -v "$ROOT/data/inference:/app/output" \
    -v "$ROOT/inference/log:/app/log" \
    -e INPUT_DIR=/app/input \
    -e OUTPUT_DIR=/app/output \
    -e LOG_DIR=/app/log \
    -e PIPELINE_STAGE=INFER \
    -e ARCH=vit_tiny \
    -e PATCH_SZ=16 \
    -e DOWNSAMPLE_SIZE=5000 \
    echoflow-infer python main.py
