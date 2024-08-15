# %%
from pathlib import Path
import xarray as xr
import os
import logging

# import echopype as ep

input_dir = os.getenv("INPUT_DIR", "/data/sonar")
output_dir = os.getenv("OUTPUT_DIR", "/data/processed")
log = os.getenv("LOG_DIR", "log")
# sonar_model = os.getenv("SONAR_MODEL","ek80")

logging.basicConfig(
    filename=Path(log) / "raw.log",
    level=logging.DEBUG,
    filemode="w",
    format="%(asctime)s - %(message)s",
)
import functools


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error occurred in function {func.__name__}: {str(e)}")
            raise

    return wrapper


from echolab2.instruments import EK80

import numpy as np


@log_errors
def read_raw(fp: Path):
    ek80 = EK80.EK80()
    logging.info(f"Reading {fp}")
    # ek80.read_idx(fp.with_suffix(".idx"))
    ek80.read_raw(fp)

    logging.info("Getting channel data")
    channel_data = ek80.get_channel_data()
    logging.debug("Getting channel data complete")

    return ek80, channel_data

@log_errors
def read_raw_ep(fp: Path):

    ek = ep.open_raw(fp,sonar_model=sonar_model)
    return ek

@log_errors
def sv_to_xarray(sv):
    logging.debug("Converting sv to xarray")
    frequency = [sv.frequency]
    ping_time = sv.ping_time
    depth = sv.depth
    sv = sv.data[None, :, :]

    return xr.DataArray(
        sv,
        coords=[frequency, ping_time, depth],
        dims=["frequency", "ping_time", "depth"],
    )


@log_errors
def generate_freq_sv_ds(fp: Path):
    _, data = read_raw(fp)
    channel_da = []

    for channel in data:
        print(channel, data[channel])
        if len(data[channel]) == 0:
            continue
        channel_obj = data[channel][0]
        sv = channel_obj.get_sv(return_depth=True)
        xr_sv = sv_to_xarray(
            sv,
        )
        channel_da.append(xr_sv)

    da = xr.concat(channel_da, dim="frequency")

    return da




def reduce_files_to_diff(inp, out):
    # check diff of output dir and input dir files.
    in_files = {f.stem for f in inp.glob("*.raw")}
    out_files = {f.stem for f in out.glob("*.nc")}
    diff = in_files - out_files
    print(diff)

    return filter(lambda x: x.stem in diff, inp.glob("*.raw"))


@log_errors
def consume_dir(input_dir: Path, output_dir: Path):
    data = None 
    files_to_compute = reduce_files_to_diff(input_dir, output_dir)
    for file in files_to_compute:
        # file = (input_dir / file).with_suffix(".raw")
        logging.debug(f"Processing file {file}")
        try:
            data = generate_freq_sv_ds(file)
        except Exception as e:
            logging.debug(e)
            continue
        try:
            data.to_netcdf((output_dir / file.stem).with_suffix(".nc"))
        except Exception as e:
            logging.debug(e)
            continue
    print("Done")
    if data is None:
        logging.error("No data was processed")

    return data


if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))
