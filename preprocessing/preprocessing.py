from pathlib import Path
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr

input_dir = os.getenv("INPUT_DIR", "/data/processed")
output_dir = os.getenv("OUTPUT_DIR", "/data/test_imgs")
log = os.getenv("LOG_DIR", ".")

def reduce_files_to_diff(inp, out):
    # Check difference of files in input and output directories
    in_files = {f.stem for f in inp.glob("*.nc")}
    out_files = {f.stem for f in out.glob("*")}
    diff = in_files - out_files
    print(diff)

    return filter(lambda x: x.stem in diff, inp.glob("*.nc"))

def process_seafloor(ds: xr.Dataset, depth0=25, backstep=5):
    # Tag the data with 'bottom detection v1' and parameters
    ds.attrs["tag"] = f"bd2-d%i-bs%i" % (depth0, backstep)

    # Drop the top of the water column ('forward step')
    ds = ds.where(ds.depth >= depth0, drop=True)

    # Compute first approximation of the bottom as the max Sv depth
    max_depth = ds.Sv.idxmax("depth").compute()
    bottom_median = max_depth.median()

    # Crop out all depths below 1.15 * bottom_median
    ds = ds.where(ds.depth <= 1.15 * bottom_median, drop=True)

    # Recompute the max depth, which represents the bottom
    bottom_depth = ds.Sv.idxmax("depth").compute()

    # Add bottom depth to the dataset for consistency
    ds["bottom_depth"] = bottom_depth

    return ds, bottom_depth

def to_colors(sv, vmin=-80, vmax=-30):
    norm = clr.Normalize(vmin=vmin, vmax=vmax, clip=True)
    y = norm(sv)
    return y * 255

def sv_to_jpg(file, vmin=-80, vmax=-30, estimate_bot=False):
    ds = xr.open_dataset(file)  # Using open_dataset to handle multiple variables
    base_out = Path(output_dir)

    for freq in ds.frequency:
        freq_data = ds.Sv.sel(frequency=freq).dropna(dim='depth')
        freq_data = 10 * np.log10(freq_data)

        # Check if the resulting frequency data is empty or invalid
        if freq_data.size == 0:
            print(f"Warning: No valid Sv data for frequency {freq.data} in file {file.stem}. Skipping.")
            continue

        if "bottom_depth" in ds:
            offset = 3
            # Select bottom depth for the specific frequency to avoid dimension mismatch
            bottom_depth = ds["bottom_depth"].sel(frequency=freq).dropna(dim='ping_time')
            # Apply filtering based on bottom_depth for the selected frequency
            freq_data = freq_data.where(freq_data.depth <= bottom_depth + offset, drop=True)
            freq_data = freq_data.where(freq_data.depth >= 25, drop=True)
        elif estimate_bot:
            ds, bottom_depth = process_seafloor(ds)
            freq_data = ds.Sv.sel(frequency=freq).dropna(dim='depth')

            # bottom_depth_for_freq = bottom_depth.sel(frequency=freq).dropna(dim='ping_time')

            # freq_data = freq_data.where(freq_data.depth <= bottom_depth_for_freq + 3, drop=True)
            # freq_data = freq_data.where(freq_data.depth >= 25, drop=True)

        sv = np.array(freq_data.data)

        if sv.size == 0 or np.isnan(sv).all():
            print(f"Warning: Empty or invalid Sv data for frequency {freq.data} in file {file.stem}. Skipping.")
            continue

        sv_colors = to_colors(sv, vmin, vmax)
        
        # Convert to an image
        plt.imshow(sv_colors.T, aspect='auto')
        plt.savefig(f"./{str(freq)}.jpg")
        plt.clf()
        img = Image.fromarray(sv_colors.astype(np.uint8))
        img = img.convert("L")

        print(f"Saving image for frequency {freq.data} with shape {img.size}")

        save_path = base_out / file.stem
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure that the image is not empty
        if img.size[0] == 0 or img.size[1] == 0:
            print(f"Warning: Generated an empty image for frequency {freq.data}. Skipping.")
            continue

        img.save(save_path / f"{int(freq.data)}.jpg")

def consume_dir(input_dir: Path, output_dir: Path):
    print(list(input_dir.glob("*")))
    files_to_consume = reduce_files_to_diff(input_dir, output_dir)
    for file in files_to_consume:
        sv_to_jpg(file, estimate_bot=True)

    return None

if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))