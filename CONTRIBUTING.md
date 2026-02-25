# Contributing to EchoFlow

Thank you for your interest in contributing to EchoFlow! This document explains how to report bugs, suggest features, and submit code changes.
Contributions such as:
- Generalizing reader to [EchoPype](https://echopype.readthedocs.io/en/latest/) will allow for generic echosounder data support.

## Reporting bugs

Please open a [GitHub Issue](https://github.com/erlingdevold/EchoFlow/issues) and include:

- A clear, descriptive title
- Steps to reproduce the problem
- Expected behaviour vs. actual behaviour
- Your environment (OS, Docker version, Docker Compose version)
- Relevant log output (from `docker compose logs` or the stage log directories)

## Suggesting features

Open a GitHub Issue with the label **enhancement** and describe:

- The use case or problem the feature addresses
- A sketch of how you imagine it working
- Any alternative approaches you considered

## Development setup

1. Clone the repository with submodules:
   ```bash
   git clone --recurse-submodules https://github.com/erlingdevold/EchoFlow.git
   cd EchoFlow
   ```

2. Install per-stage dependencies (work on a single stage or all three):
   ```bash
   pip install -r raw_consumer/requirements.txt
   pip install -r preprocessing/requirements.txt
   pip install -r inference/requirements.txt
   ```

3. Populate the test input (requires AWS CLI):
   ```bash
   aws s3 cp --no-sign-request \
     "s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw" \
     data/input/
   touch ./inference/checkpoint.pth
   git submodule sync --recursive
   ```

4. Build and run the pipeline locally:
   ```bash
   docker compose up --build
   ```

## Pull request workflow

1. **Branch naming**: use a short, descriptive name prefixed by the type of change, e.g. `fix/monitor-volumes`, `feat/slurm-support`, `docs/contributing-guide`.
2. **Commit messages**: write clear, imperative-mood messages (`Fix monitor volume mounts`, not `fixed stuff`).
3. **Tests**: the CI pipeline runs automatically on every push and pull request. Ensure your changes do not break existing tests before opening a PR.
4. **PR description**: reference any related issues (e.g. `Closes #5`) and summarise the change and why it is needed.
5. **Review**: at least one approving review is required before merging to `master`.

## Code of conduct

This project follows the [GitHub Community Code of Conduct](https://docs.github.com/en/site-policy/github-terms/github-community-code-of-conduct). By participating you agree to abide by its terms.
