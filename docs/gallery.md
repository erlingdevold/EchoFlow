# Gallery

Example outputs from the EchoFlow pipeline processing a NOAA EK80 test file (`Hake-D20230811-T165727.raw`).

## Preprocessed echogram (38 kHz)

![Preprocessed 38 kHz echogram](figures/example.png)

The echogram shows volume-backscattering strength after contrast stretching, seafloor detection, and sigma thresholding.

## Pipeline overview

The figure below shows a preprocessed echogram (left) alongside the DINO attention map (right), illustrating how the Vision Transformer highlights salient acoustic structures such as fish schools and seabed returns.

![Echogram and attention map](figures/example.png)
