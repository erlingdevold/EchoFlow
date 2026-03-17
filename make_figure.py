# %%
"""Generate a clean paper figure from pipeline outputs."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# %%
data_dir = Path("data")
sample = "Hake-D20230811-T165727"
sample_dir = data_dir / "preprocessing" / sample
figures_dir = Path("figures")
figures_dir.mkdir(exist_ok=True)

frequencies = sorted(int(p.stem) for p in sample_dir.glob("*.png") if p.stem.isdigit())
print(f"Frequencies: {frequencies}")

# %%
import torch
import torchvision.transforms as pth_transforms
import torch.nn as nn
import sys
sys.path.insert(0, "inference")
import vision_transformer as vits

patch_size = 16
# image_size is (height, width) and must be divisible by patch_size
# Original: depth=11493, time=213 → downsample preserving aspect ratio
image_size = (2400, 224)  # (height=depth, width=time)
image_size = (
    (image_size[0] // patch_size) * patch_size,
    (image_size[1] // patch_size) * patch_size,
)

device = torch.device("cpu")
model = vits.__dict__["vit_small"](patch_size=patch_size, num_classes=0).eval().to(device)
url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
state_dict = torch.hub.load_state_dict_from_url(
    url=f"https://dl.fbaipublicfiles.com/dino/{url}", map_location="cpu", weights_only=False
)
model.load_state_dict(state_dict, strict=False)
model.eval()
print("Model loaded")

# %%
transform = pth_transforms.Compose([
    pth_transforms.Resize(image_size),
    pth_transforms.ToTensor(),
    pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

for freq in frequencies:
    print(f"\n[{freq} Hz] Processing...")
    echogram_path = sample_dir / f"{freq}.png"

    # Load and transpose: (ping_time, depth) → (depth, time)
    echo_img = np.array(Image.open(echogram_path)).T

    # Apply mask to zero out seafloor before inference
    # The raw mask has holes — compute a clean per-ping bottom depth and mask below it
    mask_path = sample_dir / f"{freq}_mask.npy"
    raw_mask = np.load(mask_path).T  # (depth, time), True = valid water column

    # For each time column, find the last valid depth index
    n_depth, n_time = raw_mask.shape
    bottom_idx = np.full(n_time, n_depth, dtype=int)
    for t in range(n_time):
        valid = np.where(raw_mask[:, t])[0]
        if len(valid) > 0:
            bottom_idx[t] = valid[-1]

    # Smooth the bottom line to remove single-ping jumps
    from scipy.ndimage import median_filter
    bottom_idx = median_filter(bottom_idx, size=5)

    # Build a clean filled mask: True for all depths above the bottom line
    mask = np.zeros_like(raw_mask)
    for t in range(n_time):
        mask[:bottom_idx[t], t] = True

    masked_img = np.where(mask, echo_img, 0).astype(np.uint8)

    # Prepare image tensor from masked echogram
    img_pil = Image.fromarray(masked_img).convert("RGB")
    img_tensor = transform(img_pil).unsqueeze(0)

    # Get attention maps
    with torch.no_grad():
        attentions = model.get_last_selfattention(img_tensor)[0, :, 0, 1:]
    nh = attentions.shape[0]
    h, w = image_size[0] // patch_size, image_size[1] // patch_size
    attentions = attentions.reshape(nh, h, w)
    attentions = nn.functional.interpolate(
        attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest"
    )[0].detach().cpu().numpy()

    for i in range(nh):
        a = attentions[i]
        attentions[i] = (a - a.min()) / (a.max() - a.min() + 1e-8)

    mean_att = attentions.mean(axis=0)

    # Resize echogram to match attention map dimensions
    echo_resized = np.array(Image.fromarray(echo_img).resize((image_size[1], image_size[0])))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(6, 10), gridspec_kw={"wspace": 0.3})

    im0 = axes[0].imshow(echo_resized, cmap="viridis", aspect="auto")
    axes[0].set_xlabel("Ping time")
    axes[0].set_ylabel("Depth")
    axes[0].set_title(f"{freq // 1000} kHz Echogram", fontsize=11)

    im1 = axes[1].imshow(mean_att, cmap="inferno", aspect="auto")
    axes[1].set_xlabel("Ping time")
    axes[1].set_title("DINO mean self-attention", fontsize=11)

    for ax in axes:
        ax.set_yticks([])
        ax.set_xticks([])

    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04, label="Sv (normalised)")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04, label="Attention weight")

    plt.savefig(figures_dir / f"example_{freq}.png", dpi=200, bbox_inches="tight")
    plt.savefig(figures_dir / f"example_{freq}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved example_{freq}.png")
