---
title: "EchoFlow – End-to-end self-supervised sonar-image pipeline"
tags:
  - computer-vision
  - self-supervised-learning
  - vision-transformer
  - marine-robotics
  - fisheries-acoustics
authors:
  - name: Erling Devold
    orcid: 0009-0000-0949-5992
    affiliation: 1
affiliations:
  - name: SINTEF AS, Trondheim, Norway
    index: 1
date: 2025-06-10
doi: 10.5281/zenodo.XXXXXXX  # ← replace with real DOI after Zenodo deposit
---


## Summary
**EchoFlow** is a three-stage, containerised workflow that turns raw Kongsberg EK80
sonar files into human-readable images and machine-interpretable attention maps:

1. **Conversion** – decode `.raw` pings, calibrate backscatter with *pyEcholab* [1], and store the result as NetCDF cubes.
2. **Pre-processing** – contrast-stretch and down-sample echograms as PNG slices.  
3. **Inference** – apply a self-supervised Vision Transformer (DINO) to generate
   per-patch attention heatmaps that highlight salient regions such as fish schools
   or seabed returns.

The whole pipeline is orchestrated by Docker Compose and verified on commit
with GitHub Actions.

## Statement of need
Marine-acoustics laboratories collect terabytes of sonar each survey but lack a
single open-source toolchain that

* converts multi-frequency data,  
* scales from a laptop to an HPC cluster, and  
* integrates state‑of‑the‑art computer‑vision models.  

Existing libraries such as *echopype*  Lee et al. [2] focus on the
conversion step. While useful for handling raw data, it lacks inference capabilities suitable for deployment on unmanned vehicles.
EchoFlow fills this gap by chaining conversion → pre-processing → 
self-supervised inference in one reproducible workflow, enabling rapid analysis, quick modularization and Bring-Your-Own Loop

## Implementation and architecture
Each stage lives in its own Docker image and shares data through volume mounts
(`./data/<stage>`).  A Python watchdog monitors the input directory and triggers
processing when new files land; continuous‑integration builds the Docker Compose stack and asserts that at least one attention map is produced. 
Pre-trained DINO weights are downloaded on first use and cached for subsequent runs.

## Illustrative example
```bash
# 0 - get prerequisites
aws s3 cp --no-sign-request "s3://noaa-wcsd-pds/data/raw/Bell_M._Shimada/SH2306/EK80/Hake-D20230811-T165727.raw" data/input
touch ./inference/checkpoint.pth
git submodule sync
# 1 – launch the complete pipeline
docker compose up --build raw preprocessing infer

# 2 – first output echogram and its attention map
xdg-open data/inference/Hake-D20230811-T165727/70000.png
xdg-open data/preprocessing/Hake-D20230811-T165727/38000_debug.jpg

```
## Conclusion
EchoFlow packages conversion, pre‑processing, and state‑of‑the‑art
self‑supervised inference for echosounder data into a reproducible,
container‑based workflow. By automating the path from raw sonar to
attention‑based visual analytics it enables fisheries scientists and
marine‑robotics engineers to explore large datasets without proprietary
software or labelled images. The project is fully open source,
continuously tested, and ready for community extension.

## References
[1] J. Sullivan, D. Chu, and W.-J. Lee, “PyEcholab: An open‑source Python‑based toolkit to process and visualize echosounder data,” *Journal of the Acoustical Society of America*, vol. 144, no. 3, p. 1778, Sep. 2018, doi: 10.1121/1.5053981.

[2] W.-J. Lee, L. Setiawan, C. Tuguinay, E. Mayorga, and V. Staneva, “Interoperable and scalable echosounder data processing with Echopype,” *ICES Journal of Marine Science*, vol. 181, no. 10, pp. 1941–1951, Dec. 2024, doi: 10.1093/icesjms/fsae133.