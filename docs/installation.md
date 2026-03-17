# Installation

## Prerequisites

- Docker ≥ 24
- Docker Compose v2 (`docker compose` — note: no hyphen)
- Git
- AWS CLI (for downloading the sample input file)

## Clone with submodules

```bash
git clone --recurse-submodules https://github.com/erlingdevold/EchoFlow.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

## Docker (recommended)

No additional installation needed — all dependencies are bundled in the Docker images.

```bash
docker compose up --build
```

## Native install (without Docker)

The raw_consumer stage requires HDF5 and netCDF C libraries:

=== "macOS"

    ```bash
    brew install hdf5 netcdf
    ```

=== "Ubuntu / Debian"

    ```bash
    apt-get install libhdf5-dev libnetcdf-dev
    ```

Then install Python dependencies:

```bash
pip install -r raw_consumer/requirements.txt
pip install -r preprocessing/requirements.txt
pip install -r inference/requirements.txt
```
