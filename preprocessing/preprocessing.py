# placeholder file


from pathlib import Path
import os
import xarray as xr
import numpy as np

input_dir = os.getenv("INPUT_DIR", "/data/processed")
output_dir = os.getenv("OUTPUT_DIR", "/data/test_imgs")
log = os.getenv("LOG_DIR", ".")

def reduce_files_to_diff(inp, out):
    # check diff of output dir and input dir files.
    in_files = {f.stem for f in inp.glob("*.nc")}
    out_files = {f.stem for f in out.glob("*.npy")}
    diff = in_files - out_files
    print(diff)

    return filter(lambda x: x.stem in diff, inp.glob("*.nc"))

def process_seafloor(da: xr.DataArray, depth0=25, backstep=5, return_idx=True):
    # tag the data with 'bottom detection v1' and parameters
    da.attrs["tag"] = f"bd2-d%i-bs%i" % (depth0, backstep)

    # drop the top of the water column ('forward step')
    da = da.where(da.depth >= depth0, drop=True)

    # compute first approx. of the bottom as the max sv depth
    max_depth = da.idxmax("depth").compute()
    bottom_median = max_depth.median()

    # crop out all depths below 1.8 * bottom_median
    da = da.where(da.depth <= 1.15 * bottom_median, drop=True)

    # recompute the max
    max_depth = da.idxmax("depth").compute()
    total_max = max_depth.max()

    # da = xr.where(da.depth <= max_depth - backstep, da, np.nan, keep_attrs=True)

    return da, max_depth
    return da.where(da.depth <= total_max, drop=True), max_depth

main_frequency = 0.38e5
import matplotlib.colors as clr
def to_colors(sv, vmin=-80, vmax=-30):
    norm = clr.Normalize(vmin=vmin, vmax=vmax, clip=True)
    y = norm(sv)
    return y * 255

import matplotlib.pyplot as plt
def sv_to_jpg(file,vmin=-80,vmax=-30):
    
    da = xr.open_dataarray(file)
    # ppda, max_array = process_seafloor(da)
    # print(ppda.mean())
    
        # sv = sv.astype(np.uint8)
    
    for freq in da:
        print(freq.shape)
        # sv = np.array(ppda.data)
        # sv = to_colors(sv)
        freq = 10 * np.log10(freq)
        print(freq.mean())
        plt.imshow(freq,aspect='auto',vmin = -80,vmax=-30)
        plt.show() # white plot whaiiiiii
    # img = Image.fromarray(sv)
    


def consume_dir(input_dir: Path, output_dir: Path):
    print(list(input_dir.glob("*")))
    files_to_consume = reduce_files_to_diff(input_dir, output_dir)
    for file in files_to_consume:
        sv_to_jpg(file)
        
    return None


if __name__ == "__main__":

    consume_dir(Path(input_dir), Path(output_dir))