#!/bin/bash
docker build . -f Dockerfile.raw -t raw
docker build . -f Dockerfile.preprocessing -t preprocessing
docker build . -f Dockerfile.infer -t infer
docker run  -v ./inp:/app/input -v ./out:/app/output -v ./log:/app/log -e INPUT_DIR=/app/input -e OUTPUT_DIR=/app/output -e LOG_DIR=/app/log raw 