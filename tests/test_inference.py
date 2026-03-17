from pathlib import Path

import pytest
import torch

# Ensure inference/ is on path (conftest.py handles this)
import inspect_attention as ia


def test_process_image_shape(synthetic_png):
    """Output tensor should be (3, H, W)."""
    image_size = (64, 64)
    tensor = ia.process_image(synthetic_png, image_size=image_size)
    assert tensor.shape == (3, 64, 64)


def test_process_image_normalized(synthetic_png):
    """Output tensor values should be roughly normalized (not [0,255])."""
    tensor = ia.process_image(synthetic_png, image_size=(32, 32))
    assert tensor.min() < 0 or tensor.max() < 10  # normalized, not raw pixel values


def test_assert_img_sizes_even(synthetic_png):
    """assert_img_sizes should return even dimensions."""
    w, h = ia.assert_img_sizes(synthetic_png, image_size=(1000, 1000))
    assert w % 2 == 0
    assert h % 2 == 0


def test_assert_img_sizes_clamps(synthetic_png):
    """Dimensions should be clamped to image_size when image is smaller."""
    w, h = ia.assert_img_sizes(synthetic_png, image_size=(1000, 1000))
    # Our synthetic image is 50x100, both smaller than 1000
    assert w <= 1000
    assert h <= 1000


@pytest.fixture(scope="module")
def dino_model():
    """Load DINO model once for all inference tests.

    Uses hub fallback: touch checkpoint.pth triggers the except branch
    which downloads from Facebook's DINO weights.
    """
    # Create a dummy checkpoint to trigger the hub fallback path
    chkp = Path("checkpoint.pth")
    chkp.touch(exist_ok=True)

    device, model = ia.setup_device_and_model(
        arch="vit_small", patch_size=16, pretrained_weights=str(chkp)
    )
    return device, model


def test_model_loaded_once(dino_model):
    """Model should be returned as an argument, not reloaded."""
    device, model = dino_model
    assert model is not None
    # Model should be in eval mode
    assert not model.training


def test_get_attention_maps_shape(dino_model, synthetic_png):
    """Attention maps should have shape (num_heads, H, W) with values in [0, 1]."""
    device, model = dino_model
    image_size = (128, 128)
    patch_size = 16

    tensor = ia.process_image(synthetic_png, image_size=image_size)
    tensor = tensor.unsqueeze(0).to(device)

    attentions = ia.get_attention_maps(
        model, tensor, image_size=image_size, patch_size=patch_size
    )

    assert attentions.ndim == 3  # (num_heads, H, W)
    assert attentions.shape[1] == image_size[0]
    assert attentions.shape[2] == image_size[1]
    assert attentions.min() >= 0.0
    assert attentions.max() <= 1.0


def test_get_attention_maps_batched(dino_model, synthetic_png):
    """Batched input should return a list of attention maps."""
    device, model = dino_model
    image_size = (128, 128)
    patch_size = 16

    tensor = ia.process_image(synthetic_png, image_size=image_size)
    batch = torch.stack([tensor, tensor]).to(device)

    results = ia.get_attention_maps(
        model, batch, image_size=image_size, patch_size=patch_size
    )

    assert isinstance(results, list)
    assert len(results) == 2
    for attn in results:
        assert attn.shape[1] == image_size[0]
        assert attn.shape[2] == image_size[1]


def test_reduce_files_to_diff(tmp_path):
    """Should identify folders that still need processing."""
    inp = tmp_path / "input"
    out = tmp_path / "output"
    inp.mkdir()
    out.mkdir()

    # Create input folders with PNGs
    (inp / "sample1").mkdir()
    (inp / "sample1" / "38000.png").touch()
    (inp / "sample2").mkdir()
    (inp / "sample2" / "70000.png").touch()

    # sample1 already processed (has output with PNG)
    (out / "sample1").mkdir()
    (out / "sample1" / "38000.png").touch()

    result = ia.reduce_files_to_diff(inp, out)
    stems = {p.stem for p in result}
    assert "sample2" in stems
    assert "sample1" not in stems
