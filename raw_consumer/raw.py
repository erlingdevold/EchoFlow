from pathlib import Path
import xarray as xr
import os
import logging
import functools
from echolab2.instruments import EK80, EK60
import numpy as np

# Directory and logging setup
input_dir = os.getenv("INPUT_DIR", "/data/sonar")
output_dir = os.getenv("OUTPUT_DIR", "/data/processed")
log = os.getenv("LOG_DIR", "log")

logging.basicConfig(
    filename=Path(log) / "raw.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(message)s",
)

# Error handling decorator
def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error occurred in function {func.__name__}: {str(e)}")
            raise
    return wrapper

@log_errors
def read_raw(fp: Path):
    if "2019847-D20190502-T004037" in str(fp): # test file for bot files
        echo = EK60.EK60()  
    else:
        echo = EK80.EK80()  
        
    logging.info(f"Reading {fp}")

    # Reading raw file
    echo.read_raw(str(fp))
    
    # Reading bot file if it exists
    bot_fp = fp.with_suffix('.bot')
    if bot_fp.exists():
        logging.info(f"Reading bot file {bot_fp}")
        echo.read_bot(str(bot_fp))
    
    # Getting channel data
    logging.info("Getting channel data")
    channel_data = echo.get_channel_data()
    logging.debug("Channel data retrieval complete")

    return echo, channel_data

@log_errors
def sv_to_xarray(sv):
    logging.debug("Converting sv to xarray")
    frequency = [sv.frequency]
    ping_time = sv.ping_time
    depth = sv.depth
    sv_data = sv.data[None, :, :]  # Reshape to add frequency dimension

    return xr.DataArray(
        sv_data,
        coords=[frequency, ping_time, depth],
        dims=["frequency", "ping_time", "depth"],
    )

@log_errors
def generate_freq_sv_ds(fp: Path):
    ek80, data = read_raw(fp)
    channel_da = []
    bottom_da = []

    # Process each channel
    for channel in data:
        print(channel, data[channel])
        if len(data[channel]) == 0:
            continue
        
        channel_obj = data[channel][0]

        # Get Sv data and convert to xarray
        sv = channel_obj.get_sv(return_depth=True)
        xr_sv = sv_to_xarray(sv)
        channel_da.append(xr_sv)

        # Get bottom data if available
        try:
            bottom = channel_obj.get_bottom(return_depth=True)
            if bottom is not None and bottom.data.size > 0:
                bot_da = xr.DataArray(
                    bottom.data,
                    coords=[sv.ping_time],
                    dims=["ping_time"],
                    name="bottom_depth"
                )
                bottom_da.append(bot_da)
        except Exception as e:
            logging.warning(f"No bottom data found for channel {channel}: {e}")

    # Concatenate the Sv data arrays across frequency dimension
    da = xr.concat(channel_da, dim="frequency")

    # Check if bottom data exists for any channels and concatenate it if it does
    if bottom_da:
        bottom_concat = xr.concat(bottom_da, dim="frequency")
        ds = xr.Dataset(
            {
                "Sv": da,
                "bottom_depth": bottom_concat
            }
        )
    else:
        ds = xr.Dataset({"Sv": da})

    return ds

def reduce_files_to_diff(inp, out):
    # Check difference of files in input and output directories
    in_files = {f.stem for f in inp.glob("*.raw")}
    out_files = {f.stem for f in out.glob("*.nc")}
    diff = in_files - out_files
    print(f"Files to process: {diff}")

    return filter(lambda x: x.stem in diff, inp.glob("*.raw"))

@log_errors
def consume_dir(input_dir: Path, output_dir: Path):
    data = None 
    files_to_compute = reduce_files_to_diff(input_dir, output_dir)
    
    for file in files_to_compute:
        logging.debug(f"Processing file {file}")
        try:
            data = generate_freq_sv_ds(file)
        except Exception as e:
            logging.error(f"Error generating dataset for {file}: {e}")
            continue
        
        try:
            data.to_netcdf((output_dir / file.stem).with_suffix(".nc"))
        except Exception as e:
            logging.error(f"Error saving dataset to NetCDF for {file}: {e}")
            continue

    print("Processing complete.")
    if data is None:
        logging.error("No data was processed")

    return data

if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))