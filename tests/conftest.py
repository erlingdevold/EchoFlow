import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

# Make module imports work without installing packages
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "inference"))
sys.path.insert(0, str(ROOT / "preprocessing"))
sys.path.insert(0, str(ROOT / "raw_consumer"))


@pytest.fixture
def synthetic_nc(tmp_path):
    """Create a synthetic netCDF file with 2 frequencies, 50 pings, 100 depth bins."""
    freqs = [38000.0, 70000.0]
    n_pings = 50
    n_depths = 100

    ping_time = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "s") for i in range(n_pings)]
    )
    depth = np.linspace(0, 500, n_depths)

    rng = np.random.default_rng(42)
    sv_linear = 10 ** (rng.uniform(-80, -30, size=(2, n_pings, n_depths)) / 10)

    sv = xr.DataArray(
        sv_linear,
        coords=[freqs, ping_time, depth],
        dims=["frequency", "ping_time", "depth"],
    )
    ds = xr.Dataset({"Sv": sv})

    nc_path = tmp_path / "test_sample.nc"
    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def synthetic_nc_with_bottom(tmp_path):
    """Synthetic netCDF with bottom_depth variable — triggers the mask branch."""
    freqs = [38000.0, 70000.0]
    n_pings = 50
    n_depths = 100

    ping_time = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "s") for i in range(n_pings)]
    )
    depth = np.linspace(0, 500, n_depths)

    rng = np.random.default_rng(42)
    sv_linear = 10 ** (rng.uniform(-80, -30, size=(2, n_pings, n_depths)) / 10)

    sv = xr.DataArray(
        sv_linear,
        coords=[freqs, ping_time, depth],
        dims=["frequency", "ping_time", "depth"],
    )

    bottom = xr.DataArray(
        rng.uniform(200, 400, size=(2, n_pings)),
        coords=[freqs, ping_time],
        dims=["frequency", "ping_time"],
    )

    ds = xr.Dataset({"Sv": sv, "bottom_depth": bottom})
    nc_path = tmp_path / "test_with_bottom.nc"
    ds.to_netcdf(nc_path)
    return nc_path


@pytest.fixture
def synthetic_png(tmp_path):
    """Create a synthetic grayscale PNG image (100x50 pixels)."""
    from PIL import Image

    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, size=(50, 100), dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    png_path = tmp_path / "test_image.png"
    img.save(png_path)
    return png_path
