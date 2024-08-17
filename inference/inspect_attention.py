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


import numpy as np
import torch
from PIL import Image, ImageOps
import torchvision.transforms as pth_transforms
import torch.nn as nn
import matplotlib.pyplot as plt


import vision_transformer as vits
def setup_device_and_model(arch='vit_tiny', patch_size=16, pretrained_weights=None, checkpoint_key='teacher'):
    device = torch.device("cpu")
    model = vits.__dict__[arch](patch_size=patch_size, num_classes=0).eval().to(device)
    for p in model.parameters():
        p.requires_grad = False

    if pretrained_weights and os.path.isfile(pretrained_weights):
        state_dict = torch.load(pretrained_weights, map_location="cpu").get(checkpoint_key, {})
        state_dict = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f'Pretrained weights loaded from {pretrained_weights}')
    else:
        print(f"Invalid path to pretrained weights: {pretrained_weights}")
    
    return device, model

def process_image(image_path, image_size=(1000, 1000)):
    img = Image.open(image_path).convert("RGB")
    
    transform = pth_transforms.Compose([
        pth_transforms.Resize(image_size),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transform(img).unsqueeze(0)

def get_attention_maps(model, img_tensor, image_size, patch_size):
    attentions = model.get_last_selfattention(img_tensor)[0, :, 0, 1:]
    nh = attentions.shape[0]
    print(attentions.shape)
    attentions = attentions.reshape(nh, image_size[0]//patch_size, image_size[1]//patch_size)
    attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu().numpy()
    attentions = (attentions - attentions.min()) / (attentions.max() - attentions.min())
    return attentions

def visualize_attention(img, attentions, nh, image_size,output_name='img.png'):
    fig, axes = plt.subplots(nh + 1, 1, figsize=(15, 15))
    print(img.size)
    axes[0].imshow(np.array(pth_transforms.Resize(image_size)(img)).T, aspect='auto')
    for i in range(nh):
        axes[i + 1].imshow(attentions[i].T, cmap='viridis', aspect='auto')
        axes[i + 1].set_title(f'Attention Head {i}')
        axes[i + 1].axis('off')
    # plt.savefig(output_name)

def assert_img_sizes(file,image_size):
    img = Image.open(file)
    array = np.array(img)
    w,h = array.shape
    downsample_w,downsample_h = image_size
    if w > downsample_w:
        w = downsample_w
    if h > downsample_h:
        h = downsample_h
    

    return w + w % 2, h + h%2

def infer_attention_heads(file,out,arch,patch_size,chkp,image_size=(1000,1000),plot=True):
    image_size = assert_img_sizes(file,image_size) # should downsample or not
    device,model = setup_device_and_model(arch=arch,patch_size=patch_size,pretrained_weights=chkp)    
    img_tensor = process_image(file,image_size=image_size)
    attentions = get_attention_maps(model, img_tensor, image_size=image_size, patch_size=patch_size)
    visualize_attention(Image.open(file), attentions, attentions.shape[0], image_size=image_size,output_name=out)
    if plot:    
        plt.savefig(out)
    else:
        pass
        # @save nc
    
        
        

def reduce_files_to_diff(inp, out):
    in_files = {f.stem for f in inp.glob("*")}
    out_files = {f.stem for f in out.glob("*")}
    diff = in_files - out_files
    print(diff)

    return filter(lambda x: x.stem in diff, inp.glob("*"))

arch = 'vit_small'
patch_size = int(os.getenv("PATCH_SZ",8))
arch = os.getenv("ARCH",'vit_small')
size = int(os.getenv("DOWNSAMPLE_SIZE", 5000))

@log_errors
def consume_dir(input_dir: Path, output_dir: Path):
    data = None
    files_to_compute = reduce_files_to_diff(input_dir, output_dir)
    for folder in files_to_compute: # these are folders now.
        base_out = output_dir / folder.name
        base_out.mkdir(parents=True,exist_ok=True)
        for file in folder.glob("*.jpg"):
            infer_attention_heads(file,base_out / file.name, arch, patch_size,"checkpoint.pth",image_size=(size,size))

    if not data:
        logging.error("No data was processed")

    return data


if __name__ == "__main__":
    consume_dir(Path(input_dir), Path(output_dir))
