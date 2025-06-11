# DINO Pipeline
[![CI](https://github.com/erlingdevold/EchoFlow/actions/workflows/ci.yml/badge.svg)](https://github.com/erlingdevold/EchoFlow/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15634054.svg)](https://doi.org/10.5281/zenodo.15634054)


This repository contains a pipeline for processing Kongsberg EK(S)60 and 80 files into Sv images. 
It includes preprocessing and inference components, with Docker support for streamlined execution in isolated environments.

## Features

- **Preprocessing**: Prepares the data for inference.
- **Inference**: Utilizes DINO for visual inference and attention inspection.
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

## Requirements

- Docker
- Docker Compose
- AWS CLI
- Git

Alternatively, you can run the pipeline outside of Docker by installing the required Python packages from respective modules `requirements.txt`.
## Populate input
```bash
aws s3 cp --no-sign-request \
  "s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw" \
  data/input
touch ./inference/checkpoint.pth
git submodule sync --recursive
```
## Setup

### Running with Docker

1. **Build and Start the Containers**:
   
   First, ensure you have Docker and Docker Compose installed. Then run the following command to start the pipeline:

   ```bash
   docker compose up --build
   ```

   This will build the Docker containers for preprocessing and inference and start the pipeline.

2. **Running the Preprocessing**:

   Once the containers are running, the pipeline is started, given that you populated input.

This quickstart shows how easy the pipeline is setup in order to 

## ENV variables

	1.	watchdog.py:
	•	LOG_DIR (default: "/data/log")
	•	INPUT_DIR (default: "/data/sonar")
	•	OUTPUT_DIR (default: "/data/processed")
	2.	inspect_attention.py:
	•	INPUT_DIR (default: "/data/test_imgs")
	•	OUTPUT_DIR (default: "/data/inference")
	•	LOG_DIR (default: ".")
	•	PATCH_SZ (default: 8)
	•	ARCH (default: 'vit_small')
	•	DOWNSAMPLE_SIZE (default: 5000)
	3.	preprocessing.py:
	•	INPUT_DIR (default: "/data/processed")
	•	OUTPUT_DIR (default: "/data/test_imgs")
	•	LOG_DIR (default: ".")
	4.	raw.py:
	•	INPUT_DIR (default: "/data/sonar")
	•	OUTPUT_DIR (default: "/data/processed")
	•	LOG_DIR (default: "log")
	5.	segmentation.py:
	•	INPUT_DIR (default: "/data/sonar")
	•	OUTPUT_DIR (default: "/data/processed")
	•	LOG_DIR (default: "/data/logs")

## Output

The output of the inference step, including generated attention maps and transformed images, will be saved in the `inference/output/` directory. Each run will create a timestamped subdirectory for organized output management.

## License

Licensed under the MIT License – see [LICENSE](LICENSE) for details.

## Acknowledgements

This pipeline uses the DINO Vision Transformer for attention-based image analysis. The implementation is based on research from the original DINO paper by Facebook AI Research (FAIR).
