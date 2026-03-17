---
title: "EchoFlow: End-to-end self-supervised sonar-image pipeline"
tags:
  - computer-vision
  - self-supervised-learning
  - vision-transformer
  - marine-robotics
  - fisheries-acoustics
  - containerization
authors:
  - name: Erling Devold
    orcid: 0009-0000-0949-5992
    affiliation: 1
affiliations:
  - name: SINTEF AS, Trondheim, Norway
    index: 1
date: 13 June 2025
doi: 10.5281/zenodo.15634054
bibliography: paper.bib
---

# Summary

**EchoFlow** is a three-stage, containerised workflow that converts raw Kongsberg EK80 echosounder files into human-readable echograms *and* machine-interpretable attention maps.
The EK80 is the dominant wideband echosounder from Kongsberg, the leading manufacturer in fisheries acoustics; surveys with these instruments routinely produce terabytes of multi-frequency data.

1. **Conversion** – raw `.raw` pings are decoded and calibrated to volume-back-scattering strength with *PyEcholab* [@sullivan2018pyecholab].
2. **Pre-processing** – echograms are contrast-stretched, down-sampled and tiled as PNG images.
3. **Inference** – a Vision Transformer trained with DINO [@caron2021dino] yields per-patch attention heat-maps that highlight fish schools, seabed returns, and other salient structures.

Each stage is encapsulated in its own Docker image and orchestrated with Docker Compose.
Continuous Integration (CI) ensures that a test file always produces at least one attention map per echogram frequency. The test file can be inspected as artifacts from the CI actions.

# Statement of need

Marine-acoustics researchers collect **terabytes** of multi-frequency sonar data per survey but lack an open-source tool-chain that

* converts heterogeneous raw formats,
* scales from a laptop to multi-core servers, and
* integrates state-of-the-art computer-vision models.

Today the typical workflow relies on ad-hoc scripts combining PyEcholab or Echopype for format conversion, followed by manual preprocessing and separate ML tooling—or on proprietary software such as Echoview.
No single open-source pipeline bridges raw data and ML-ready outputs.

Previous work [@lee2024echopype; @sullivan2018pyecholab] addresses the first bullet; EchoFlow fills the remaining gap by chaining **conversion → pre-processing → self-supervised inference** in a single, reproducible workflow.
This lowers the barrier for fisheries scientists, marine-robotics engineers, and citizen scientists who want modern ML without bespoke pipelines. The containerised design also facilitates deployment on shared compute environments via tools such as Singularity/Apptainer. This pipeline also serves as a foundation for incorporating modern frameworks into marine science.

EchoFlow was developed as part of the research project *Robotics underneath sub-zero waters and outer space* (RCN project 328193), where it is used to process acoustic data from autonomous underwater vehicles for ocean biomass estimation.

# Implementation and architecture

Each stage lives in its own Docker image and communicates through bind-mounted volumes (`./data/`). 
A Python watchdog triggers the pipeline when new `.raw` files arrive, and pre-trained DINO weights are cached on first use. 
Images are multi-platform (`linux/amd64`, `linux/arm64`).

# Illustrative example

```bash
# 0 – fetch a sample EK80 file (NOAA public bucket)
BUCKET="s3://noaa-wcsd-pds/data/raw"
FILE="Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw"
aws s3 cp --no-sign-request "${BUCKET}/${FILE}" data/input

# 1 – run the full pipeline
docker compose up --build raw preprocessing infer

# 2 – open the resulting echogram and attention map
xdg-open data/preprocessing/Hake-D20230811-T165727/38000_debug.jpg
xdg-open data/inference/Hake-D20230811-T165727/70000.png
```

![Preprocessed 38 kHz echogram (left) and DINO attention map at 70 kHz (right) for file Hake-D20230811-T165727.\label{fig:example}](figures/example.png)

# Acknowledgements

This work was supported by the Research Council of Norway under project 328193 (*Robotics underneath sub-zero waters and outer space*, 2022–2025).

# References