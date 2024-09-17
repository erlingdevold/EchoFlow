from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import os
import time
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.colors as clr
import logging
import json

# Set up environment variables
input_dir = os.getenv("INPUT_DIR", "/data/processed")
output_dir = os.getenv("OUTPUT_DIR", "/data/test_imgs")
log_path = os.getenv("LOG_DIR", ".")

# Set up logging configuration
log_file = os.path.join(log_path, "pipeline.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger()


def reduce_files_to_diff(inp, out):
    in_files = {
        f.stem for f in inp.glob("*.nc")
        if not (f.with_suffix(".processed")).exists() and not (f.with_suffix(".failed")).exists()
    }
    out_files = {f.stem for f in out.glob("*")}
    diff = in_files - out_files
    print(diff)
    return filter(lambda x: x.stem in diff, inp.glob("*.nc"))



# Process the seafloor data
def process_seafloor(ds: xr.DataArray, depth0=25, backstep=5):
    ds.attrs["tag"] = f"bd2-d%i-bs%i" % (depth0, backstep)
    ds = ds.where(ds.depth >= depth0, drop=True)
    max_depth = ds.idxmax("depth").compute()
    bottom_median = max_depth.median()
    ds = ds.where(ds.depth <= 1.15 * bottom_median, drop=True)
    bottom_depth = ds.idxmax("depth").compute()
    ds["bottom_depth"] = bottom_depth
    return ds, bottom_depth


# Function to convert Sv data to colors
def to_colors(sv, vmin=-80, vmax=-30):
    norm = clr.Normalize(vmin=vmin, vmax=vmax, clip=True)
    y = norm(sv)
    return y * 255


# Apply a sigma threshold to the data
def sigma_thresholding_upper(data, sigma=3):
    flattened_data = data.values.flatten()
    mean = np.nanmean(flattened_data)
    std_dev = np.nanstd(flattened_data)
    upper_bound = mean + sigma * std_dev
    thresholded_data = data.where(data <= upper_bound)
    binary_mask = data <= upper_bound
    nanmean_value = np.nanmean(flattened_data)
    filled_data = thresholded_data.fillna(nanmean_value)
    return filled_data, binary_mask


# Process and save Sv data to image and mask files
def sv_to_jpg(file, vmin=-80, vmax=-30, estimate_bot=False):
    ds = xr.open_dataset(file)
    base_out = Path(output_dir)

    for freq in ds.frequency:
        freq_data = ds.Sv.sel(frequency=freq).dropna(dim="depth")
        freq_data = 10 * np.log10(freq_data)

        if freq_data.size == 0:
            log.warning(f"No valid Sv data for frequency {freq.data} in file {file.stem}. Skipping.")
            continue

        if "bottom_depth" in ds:
            offset = 3
            bottom_depth = ds["bottom_depth"].sel(frequency=freq).dropna(dim="ping_time")
            freq_data = freq_data.where(freq_data.depth <= bottom_depth + offset, drop=True)
            freq_data = freq_data.where(freq_data.depth >= 25, drop=True)
        elif estimate_bot:
            debug_path = Path(log_path)
            plt.imshow(freq_data, aspect="auto", vmin=-80, vmax=-30)
            plt.savefig(debug_path / f"{file.stem}_debug.jpg")
            plt.clf()

            try:
                freq_data, bottom_depth = process_seafloor(freq_data)
            except ValueError as e:
                log.error(f"Seafloor processing failed for file {file.stem}, frequency {freq.data}: {e}")
                mark_as_failed(file)  # Mark the file as failed
                return False  # Skip further processing for this file

            _, mask = sigma_thresholding_upper(freq_data)
            print(mask.shape)

        sv = np.array(freq_data.data)

        if sv.size == 0 or np.isnan(sv).all():
            log.warning(f"Empty or invalid Sv data for frequency {freq.data} in file {file.stem}. Skipping.")
            continue

        sv_colors = to_colors(sv, vmin, vmax)

        img = Image.fromarray(sv_colors.astype(np.uint8))
        img = img.convert("L")

        log.info(f"Saving image for frequency {freq.data} with shape {img.size}")

        save_path = base_out / file.stem
        save_path.mkdir(parents=True, exist_ok=True)

        img_file = save_path / f"{int(freq.data)}.png"
        mask_file = save_path / f"{int(freq.data)}_mask.npy"

        if img.size[0] == 0 or img.size[1] == 0:
            log.warning(f"Generated an empty image for frequency {freq.data}. Skipping.")
            continue

        plt.imshow(img, aspect="auto")
        plt.savefig(save_path / f"{int(freq.data)}_debug.jpg")

        try:
            img.save(img_file)
            np.save(mask_file, mask)

            if img_file.exists() and mask_file.exists():
                success = True
                mark_as_processed(file)  # Mark the file as processed after successful processing
            else:
                success = False
                log.error(f"Failed to save files for frequency {freq.data} in file {file.stem}. Aborting.")
                mark_as_failed(file)  # Mark the file as failed
                break

        except Exception as e:
            success = False
            log.error(f"Error saving image or mask for frequency {freq.data}: {e}")
            mark_as_failed(file)  # Mark the file as failed
            break

    return success



# Mark file as processed by deleting .nc file and creating a marker
def mark_as_processed(file: Path):
    try:
        file.unlink()
        log.info(f"Deleted file {file}")
        marker_file = file.with_suffix(".processed")
        marker_file.touch()
        log.info(f"Created marker file {marker_file}")
    except Exception as e:
        log.error(f"Error deleting file {file} or creating marker: {e}")


# Check if a file is ready to be processed
def is_file_ready(file: Path, retries=10, wait_time=1) -> bool:
    retry_count = 0
    while retry_count < retries:
        if not file.exists():
            log.warning(f"File {file} does not exist. Retry {retry_count}/{retries}")
            return False

        if file.stat().st_size == 0:
            log.warning(f"File {file} is empty. Retry {retry_count}/{retries}")
            return False

        try:
            ds = xr.open_dataset(file)
            ds.close()
            log.info(f"File {file} is ready after {retry_count} retries.")
            return True
        except Exception as e:
            log.error(f"Error opening file {file}: {e}. Retry {retry_count}/{retries}")
            time.sleep(wait_time)
            retry_count += 1

    log.error(f"File {file} is not ready after {retries} retries.")
    return False


import gc
# Mark file as failed by creating a .failed file
def mark_as_failed(file: Path):
    try:
        marker_file = file.with_suffix(".failed")
        marker_file.touch()  # Create an empty .failed file
        log.info(f"Marked file as failed: {marker_file}")
    except Exception as e:
        log.error(f"Error marking file as failed {file}: {e}")



import gc

# Process an individual file
def process_file(file: Path, output_dir: Path):
    try:
        if is_file_ready(file, retries=10, wait_time=1):
            if sv_to_jpg(file, estimate_bot=True):
                mark_as_processed(file)  # Mark file as processed
            else:
                log.error(f"Skipping file {file} due to save failure.")
                mark_as_failed(file)  # Mark file as failed
        else:
            log.error(f"Skipping file {file} after retries.")
            mark_as_failed(file)  # Mark file as failed
    except Exception as e:
        logging.error(f"Error processing {file}: {e}")
        mark_as_failed(file)  # Mark file as failed in case of any exception
    finally:
        # Force garbage collection to free memory
        gc.collect()



# Process all files in a directory
def consume_dir(input_dir: Path, output_dir: Path, max_workers=12):
    files_to_compute = list(reduce_files_to_diff(input_dir, output_dir))

    logging.info(f"Starting to process {len(files_to_compute)} files in parallel.")

    # Process files in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_file, file, output_dir): file
            for file in files_to_compute
        }

        for future in as_completed(future_to_file):
            file = future_to_file[future]
            try:
                future.result()  # This will raise an exception if one occurred during processing
            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")


# Entry point of the script
if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))
