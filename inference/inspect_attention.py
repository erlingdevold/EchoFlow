# %%
from pathlib import Path
import os
import logging


input_dir = os.getenv("INPUT_DIR", "/data/test_imgs")
output_dir = os.getenv("OUTPUT_DIR", "/data/inference")
log = os.getenv("LOG_DIR", ".")


logging.basicConfig(
    filename=Path(log) / "raw.log",
    level=logging.DEBUG,
    filemode="a",
    format="%(asctime)s - %(message)s",
)

import functools  # noqa: E402


def log_errors(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error occurred in function {func.__name__}: {str(e)}")
            raise

    return wrapper


import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
import torchvision.transforms as pth_transforms  # noqa: E402
import torch.nn as nn  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


import vision_transformer as vits  # noqa: E402

DEVICE = torch.device(
    os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 4))


def setup_device_and_model(
    arch="vit_tiny", patch_size=16, pretrained_weights=None, checkpoint_key="teacher"
):
    device = DEVICE
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    if pretrained_weights and os.path.isfile(pretrained_weights):
        try:
            state_dict = torch.load(
                pretrained_weights, map_location=device, weights_only=False
            ).get(checkpoint_key, {})
            state_dict = {
                k.replace("module.", "").replace("backbone.", ""): v
                for k, v in state_dict.items()
            }
            model.load_state_dict(state_dict, strict=False)
        except Exception:
            # pull from timm
            if patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif patch_size == 8:
                url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"  # model used for visualizations in our paper
            else:
                raise ValueError(f"Invalid patch size: {patch_size}")

            model = (
                vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0)
                .eval()
                .to(device)
            )
            state_dict = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/dino/" + url,
                map_location=device,
                weights_only=False,
            )
            model.load_state_dict(state_dict, strict=False)
            model.eval()

    return device, model


def process_image(image_path, image_size=(1000, 1000)):
    img = Image.open(image_path).convert("RGB")

    transform = pth_transforms.Compose(
        [
            pth_transforms.Resize(image_size),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transform(img)


def get_attention_maps(model, img_tensor, image_size, patch_size):
    """Extract attention maps. img_tensor can be (1,C,H,W) or (B,C,H,W)."""
    raw_attn = model.get_last_selfattention(img_tensor)
    B = raw_attn.shape[0]
    results = []
    for b in range(B):
        attn = raw_attn[b, :, 0, 1:]
        nh = attn.shape[0]
        attn = attn.reshape(
            nh, image_size[0] // patch_size, image_size[1] // patch_size
        )
        attn = (
            nn.functional.interpolate(
                attn.unsqueeze(0), scale_factor=patch_size, mode="nearest"
            )[0]
            .detach()
            .cpu()
            .numpy()
        )
        attn = (attn - attn.min()) / (attn.max() - attn.min())
        results.append(attn)
    if B == 1:
        return results[0]
    return results


def visualize_attention(img, attentions, nh, image_size):
    fig, axes = plt.subplots(nh + 1, 1, figsize=(15, 15))
    # Resize the image and convert to numpy array
    resized_img = pth_transforms.Resize(image_size)(img)
    img_array = np.array(resized_img)
    axes[0].imshow(img_array, aspect="auto")
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    # Display attention maps
    for i in range(nh):
        axes[i + 1].imshow(attentions[i], cmap="viridis", aspect="auto")
        axes[i + 1].set_title(f"Attention Head {i}")
        axes[i + 1].axis("off")

    plt.tight_layout()


def assert_img_sizes(file, image_size):
    img = Image.open(file)
    array = np.array(img)

    w, h = array.shape
    downsample_w, downsample_h = image_size
    if w > downsample_w:
        w = downsample_w
    if h > downsample_h:
        h = downsample_h

    return w + w % 2, h + h % 2


def infer_attention_heads(file, out, model, patch_size, image_size=(1000, 1000)):
    """Single-image inference (used when images have varying sizes)."""
    image_size = assert_img_sizes(file, image_size)
    img_tensor = process_image(file, image_size=image_size).unsqueeze(0).to(DEVICE)
    attentions = get_attention_maps(
        model, img_tensor, image_size=image_size, patch_size=patch_size
    )
    visualize_attention(
        Image.open(file), attentions, attentions.shape[0], image_size=image_size
    )
    plt.savefig(out)
    plt.close()


def reduce_files_to_diff(inp: Path, out: Path):
    """
    Return a list of sub-folders in *inp* that still need processing.

    * We treat each first-level folder (stem) in *inp* as one logical sample.
    * A folder counts as *already processed* if a corresponding folder
      exists in *out* **and** contains at least one PNG file.
    """
    in_stems = {p.stem for p in inp.glob("*") if p.is_dir()}
    processed_stems = {
        p.stem for p in out.glob("*") if p.is_dir() and any(p.glob("*.png"))
    }

    pending_stems = in_stems - processed_stems
    return [p for p in inp.glob("*") if p.stem in pending_stems]


patch_size = int(os.getenv("PATCH_SZ", 8))
arch = os.getenv("ARCH", "vit_small")
size = int(os.getenv("DOWNSAMPLE_SIZE", 5000))


@log_errors
def consume_dir(input_dir: Path, output_dir: Path):
    device, model = setup_device_and_model(
        arch=arch, patch_size=patch_size, pretrained_weights="checkpoint.pth"
    )
    files_to_compute = reduce_files_to_diff(input_dir, output_dir)

    # Collect all (input_path, output_path) pairs
    all_files = []
    for folder in files_to_compute:
        base_out = output_dir / folder.name
        base_out.mkdir(parents=True, exist_ok=True)
        for file in folder.glob("*.png"):
            all_files.append((file, base_out / file.name))

    if not all_files:
        logging.warning("No files to process")
        return

    image_size = (size, size)
    logging.info(
        f"Processing {len(all_files)} images in batches of {BATCH_SIZE} on {DEVICE}"
    )

    for i in range(0, len(all_files), BATCH_SIZE):
        batch = all_files[i : i + BATCH_SIZE]
        tensors = torch.stack([process_image(f, image_size) for f, _ in batch]).to(
            device
        )

        try:
            with torch.no_grad():
                attn_maps = get_attention_maps(
                    model, tensors, image_size=image_size, patch_size=patch_size
                )
        except torch.cuda.OutOfMemoryError:
            logging.warning("CUDA OOM — falling back to CPU for this batch")
            torch.cuda.empty_cache()
            model.cpu()
            device = torch.device("cpu")
            tensors = tensors.cpu()
            with torch.no_grad():
                attn_maps = get_attention_maps(
                    model, tensors, image_size=image_size, patch_size=patch_size
                )

        # Single-image batch returns a single array, not a list
        if len(batch) == 1:
            attn_maps = [attn_maps]

        for (file, out_path), attentions in zip(batch, attn_maps):
            visualize_attention(
                Image.open(file), attentions, attentions.shape[0], image_size=image_size
            )
            plt.savefig(out_path)
            plt.close()

    logging.info(f"Finished processing {len(all_files)} images")


if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))
