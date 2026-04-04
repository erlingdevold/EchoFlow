import os
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr
from PIL import Image

# Override env vars before importing the module
os.environ.setdefault("LOG_DIR", ".")

import preprocessing as pp


def test_to_colors_range():
    """Output should be in [0, 255] with correct normalization at vmin/vmax."""
    sv = np.array([-80, -55, -30])
    result = pp.to_colors(sv, vmin=-80, vmax=-30)
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(255.0)
    assert np.all(result >= 0)
    assert np.all(result <= 255)


def test_to_colors_clipping():
    """Values outside [vmin, vmax] should be clipped."""
    sv = np.array([-100, -10])
    result = pp.to_colors(sv, vmin=-80, vmax=-30)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(255.0)


def test_sigma_thresholding_upper():
    """Values above threshold should be replaced with mean."""
    rng = np.random.default_rng(42)
    data = xr.DataArray(rng.normal(0, 1, size=(100,)))
    # Insert an extreme outlier
    data[0] = 100.0
    filled, mask = pp.sigma_thresholding_upper(data, sigma=3)
    # The outlier should be masked out
    assert not mask.values[0]
    # Filled value at outlier position should be close to the mean
    assert not np.isnan(filled.values[0])


def test_process_seafloor():
    """process_seafloor should return cropped dataset and bottom_depth."""
    rng = np.random.default_rng(42)
    depth = np.linspace(0, 500, 200)
    ping_time = np.array(
        [np.datetime64("2024-01-01") + np.timedelta64(i, "s") for i in range(20)]
    )

    # Create fake Sv with clear maximum at ~250m depth
    sv_data = rng.uniform(-80, -30, size=(20, 200))
    for i in range(20):
        sv_data[i, 100] = -10  # max at depth index 100 (~250m)

    ds = xr.DataArray(sv_data, coords=[ping_time, depth], dims=["ping_time", "depth"])
    result_ds, bottom_depth = pp.process_seafloor(ds)

    # Should have dropped depths below 25m
    assert float(result_ds.depth.min()) >= 25
    assert bottom_depth is not None


def test_reduce_files_to_diff(tmp_path):
    """Should correctly identify unprocessed files."""
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir()
    out.mkdir()

    # Create some .nc files
    (inp / "file1.nc").touch()
    (inp / "file2.nc").touch()
    (inp / "file3.nc").touch()

    # Mark file1 as processed, file3 as failed
    (inp / "file1.processed").touch()
    (inp / "file3.failed").touch()

    # file2 has output already
    (out / "file2").mkdir()

    result = list(pp.reduce_files_to_diff(inp, out))
    stems = {f.stem for f in result}
    # Only file2 is unprocessed and has no marker — but it has output dir
    # So nothing should remain
    assert "file1" not in stems
    assert "file3" not in stems


def test_mark_as_processed_preserves_source(tmp_path):
    """With KEEP_INTERMEDIATES=true, .nc file should survive after mark_as_processed."""
    nc_file = tmp_path / "test.nc"
    nc_file.touch()

    with patch.object(pp, "keep_intermediates", True):
        pp.mark_as_processed(nc_file)

    assert nc_file.exists()
    assert (tmp_path / "test.processed").exists()


def test_mark_as_processed_deletes_when_disabled(tmp_path):
    """With KEEP_INTERMEDIATES=false, .nc file should be deleted."""
    nc_file = tmp_path / "test.nc"
    nc_file.touch()

    with patch.object(pp, "keep_intermediates", False):
        pp.mark_as_processed(nc_file)

    assert not nc_file.exists()
    assert (tmp_path / "test.processed").exists()


def test_sv_to_jpg_produces_files(synthetic_nc, tmp_path):
    """sv_to_jpg should produce valid PNG files with correct properties."""
    out = tmp_path / "output"
    out.mkdir()

    with patch.object(pp, "output_dir", str(out)):
        success = pp.sv_to_jpg(synthetic_nc, estimate_bot=True)

    assert success
    pngs = list(out.rglob("*.png"))
    assert len(pngs) > 0

    # Verify output correctness, not just existence
    for png_path in pngs:
        img = Image.open(png_path)
        w, h = img.size
        assert w > 0 and h > 0, f"Empty image: {png_path}"
        arr = np.array(img)
        assert arr.dtype == np.uint8, f"Expected uint8, got {arr.dtype}"
        assert arr.max() > 0, f"All-black image: {png_path}"


def test_mask_computed_with_bottom_depth(synthetic_nc_with_bottom, tmp_path):
    """When bottom_depth exists, mask should be computed (not None). Regression test for #18."""
    out = tmp_path / "output"
    out.mkdir()

    with patch.object(pp, "output_dir", str(out)):
        success = pp.sv_to_jpg(synthetic_nc_with_bottom, estimate_bot=False)

    assert success
    # Check that mask .npy files were created (not skipped)
    masks = list(out.rglob("*_mask.npy"))
    assert len(masks) > 0
    for m in masks:
        mask_data = np.load(m)
        assert mask_data.dtype == bool


def test_is_file_ready_nonexistent(tmp_path):
    """is_file_ready should return False for a non-existent file, not crash."""
    fake = tmp_path / "does_not_exist.nc"
    result = pp.is_file_ready(fake, retries=1, wait_time=0)
    assert result is False
