"""Tests for raw_consumer/raw.py.

Note: Most raw.py functions depend on pyEcholab (a git submodule, not pip-installable),
so we mock that import and test the pure-logic functions.
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import xarray as xr

# Mock echolab2 before importing raw.py, since it's a git submodule
_echolab_mock = MagicMock()
sys.modules["echolab2"] = _echolab_mock
sys.modules["echolab2.instruments"] = _echolab_mock.instruments
sys.modules["echolab2.instruments.EK80"] = _echolab_mock.instruments.EK80
sys.modules["echolab2.instruments.EK60"] = _echolab_mock.instruments.EK60

import raw as raw_module  # noqa: E402


class MockSv:
    """Minimal mock of an echolab Sv object for testing sv_to_xarray."""

    def __init__(self, frequency, ping_time, depth, data):
        self.frequency = frequency
        self.ping_time = ping_time
        self.depth = depth
        self.data = data


def test_sv_to_xarray():
    """sv_to_xarray should produce correct dims and coords."""
    n_pings = 10
    n_depths = 50
    rng = np.random.default_rng(42)

    sv = MockSv(
        frequency=38000.0,
        ping_time=np.array(
            [
                np.datetime64("2024-01-01") + np.timedelta64(i, "s")
                for i in range(n_pings)
            ]
        ),
        depth=np.linspace(0, 500, n_depths),
        data=rng.uniform(-80, -30, size=(n_pings, n_depths)),
    )

    result = raw_module.sv_to_xarray(sv)

    assert isinstance(result, xr.DataArray)
    assert set(result.dims) == {"frequency", "ping_time", "depth"}
    assert result.shape == (1, n_pings, n_depths)
    assert float(result.frequency[0]) == 38000.0


def test_reduce_files_to_diff(tmp_path):
    """Should skip files that already have .nc or .processed output."""
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir()
    out.mkdir()

    # Create input .raw files
    (inp / "file1.raw").touch()
    (inp / "file2.raw").touch()
    (inp / "file3.raw").touch()

    # file1 already has .nc output
    (out / "file1.nc").touch()
    # file2 already has .processed marker
    (out / "file2.processed").touch()

    result = list(raw_module.reduce_files_to_diff(inp, out))
    stems = {f.stem for f in result}

    assert "file3" in stems
    assert "file1" not in stems
    assert "file2" not in stems
