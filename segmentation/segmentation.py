# placeholder file


from pathlib import Path
import os
import xarray as xr
import numpy as np

input_dir = os.getenv("INPUT_DIR", "/data/sonar")
output_dir = os.getenv("OUTPUT_DIR", "/data/processed")
log = os.getenv("LOG_DIR", "/data/logs")


def reduce_files_to_diff(inp, out):
    # check diff of output dir and input dir files.
    in_files = {f.stem for f in inp.glob("*.nc")}
    out_files = {f.stem for f in out.glob("*.npy")}
    diff = in_files - out_files
    print(diff)

    return filter(lambda x: x.stem in diff, inp.glob("*.nc"))

main_frequency = 1.2e5
def tolog10(da : xr.DataArray):
    return 10 * np.log10(da)

def segment_frequency(da : xr.DataArray):
    from filtering import preprocess_bd2, find_boundaries, normalize_sv, threshold_sv
    da = tolog10(da)
    da = preprocess_bd2(da, depth0=25, backstep=5)
    da = da.dropna("depth", how="all")

    da.plot()
    pass

def segment_file(fp : Path):
    ds = xr.open_dataarray(fp, )
    if isinstance(main_frequency, float) or isinstance(main_frequency, int):
        ds_freq = ds.sel(frequency=main_frequency,method='nearest')
    elif isinstance(main_frequency, list):
        pass
    print(ds_freq)


def consume_dir(input_dir: Path, output_dir: Path):
    files_to_consume = reduce_files_to_diff(input_dir, output_dir)
    for file in files_to_consume:
        segment_file(file)
        break
        
    return None


if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))