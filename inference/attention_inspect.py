
import os
import random
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as pth_transforms
import numpy as np
from PIL import Image, ImageOps
import sys

# Import custom modules
sys.path.append('/home/erling/code/subzerospace/subzerospace/bladder/dino')
import vision_transformer as vits

def pad_to_multiple_of(img, multiple):
    pad_h, pad_w = [(multiple - s % multiple) % multiple for s in img.size[::-1]]
    return ImageOps.expand(img, (0, 0, pad_w, pad_h)), pad_h, pad_w

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

def select_random_image(frequency_dataset_dir):
    images = os.listdir(frequency_dataset_dir)
    selected_image_path = os.path.join(frequency_dataset_dir, random.choice(images))
    print(f"Selected image: {selected_image_path}")
    return selected_image_path

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

def run_infer_attention_heads(image_path,out,arch,patch_size,chkp, cache=True,show=True):
    reload = 1
    if cache and reload: 
        reload = 0 

    if not reload:
        device, model = setup_device_and_model(arch=arch, patch_size=patch_size, pretrained_weights=chkp)
        reload = 1

    img_tensor = process_image(image_path)
    attentions = get_attention_maps(model, img_tensor, image_size=(1000, 1000), patch_size=patch_size)
    visualize_attention(Image.open(image_path), attentions, attentions.shape[0], image_size=(1000, 1000),output_name=out)


if __name__ == "__main__":
    device, model = setup_device_and_model(arch='vit_tiny', patch_size=16, pretrained_weights='./checkpoint.pth')
    # image_path = select_random_image('')
    image_path = '/spin/echo/frequency_dataset/38/12738_416325_38.jpg'
    img_tensor = process_image(image_path)
    attentions = get_attention_maps(model, img_tensor, image_size=(1000, 1000), patch_size=16)
    visualize_attention(Image.open(image_path), attentions, attentions.shape[0], image_size=(1000, 1000),output_name=image_path.replace(".jpg","_attention.jpg"))

    