# Based on https://github.com/RE-N-Y/imscore/blob/main/src/imscore/preference/model.py

from importlib import resources
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from PIL import Image

def get_size(size):
    if isinstance(size, int):
        return (size, size)
    elif "height" in size and "width" in size:
        return (size["height"], size["width"])
    elif "shortest_edge" in size:
        return size["shortest_edge"]
    else:
        raise ValueError(f"Invalid size: {size}")
    
def get_image_transform(processor: AutoImageProcessor):
    config = processor.to_dict()
    # Check for 'size' in config. Some processors use 'size' as dict, others as int.
    # Also check for 'crop_size'.
    # For DINOv2, size is usually {"height": 224, "width": 224}
    
    target_size = config.get("size")
    crop_size = config.get("crop_size")
    
    # Fallback if size is not in config (unlikely for AutoImageProcessor)
    if target_size is None and "shortest_edge" in config:
        target_size = config["shortest_edge"]

    resize = T.Resize(get_size(target_size)) if config.get("do_resize") else nn.Identity()
    crop = T.CenterCrop(get_size(crop_size)) if config.get("do_center_crop") else nn.Identity()
    
    # Note on rescale_factor: 0.00392156862745098 is exactly 1/255.
    # This indicates the model expects inputs in range [0, 1] (rescaled from [0, 255]).
    # Since our input tensors are typically already in [0, 1] (divided by 255), 
    # we do NOT add a Rescale transform here to avoid double-scaling.
    
    normalise = T.Normalize(mean=processor.image_mean, std=processor.image_std) if config.get("do_normalize") else nn.Identity()

    return T.Compose([resize, crop, normalise])

class DinoScorer(torch.nn.Module):
    def __init__(self, device, model_name="facebook/dinov2-base"):
        super().__init__()
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.tform = get_image_transform(self.processor)
        self.eval()
    
    def _process(self, pixels):
        dtype = pixels.dtype
        pixels = self.tform(pixels)
        pixels = pixels.to(dtype=dtype)

        return pixels
    
    # @torch.no_grad()
    # def get_embedding(self, pixels):
    #     pixels = self._process(pixels).to(self.device)
    #     outputs = self.model(pixel_values=pixels)
    #     # Use CLS token (index 0) for DINOv2/ViT
    #     embeds = outputs.last_hidden_state[:, 0]
    #     return embeds

    # @torch.no_grad()
    # def __call__(self, pixels, prompts=None, return_img_embedding=False):
    #     # DINO does not support text prompts. 
    #     # We return the image embedding if requested or just the embedding as the "score" (vector).
    #     embeds = self.get_embedding(pixels)
    #     if return_img_embedding:
    #         return embeds, embeds
    #     return embeds

    @torch.no_grad()
    def image_similarity(self, pixels, ref_pixels, alpha=0.7):
        """
        Compute similarity between images using a weighted combination of CLS and Patch tokens.
        Args:
            pixels: Input images
            ref_pixels: Reference images
            alpha: Weight for CLS token similarity (global semantics), default 0.7
                   (1-alpha) for Patch token similarity (local structure/texture)
        """
        pixels = self._process(pixels).to(self.device)
        ref_pixels = self._process(ref_pixels).to(self.device)
        
        outputs = self.model(pixel_values=pixels)
        ref_outputs = self.model(pixel_values=ref_pixels)
        
        # 1. CLS Token Similarity (Global)
        # Shape: (Batch, 1, Hidden_Dim)
        cls_embeds = outputs.last_hidden_state[:, 0:1, :]
        ref_cls_embeds = ref_outputs.last_hidden_state[:, 0:1, :]
        
        cls_embeds = cls_embeds / cls_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_cls_embeds = ref_cls_embeds / ref_cls_embeds.norm(p=2, dim=-1, keepdim=True)
        
        cls_sim = (cls_embeds * ref_cls_embeds).sum(dim=-1).squeeze(1) # (B,)

        # 2. Patch Token Similarity (Local)
        # Shape: (Batch, Num_Patches, Hidden_Dim)
        patch_embeds = outputs.last_hidden_state[:, 1:, :]
        ref_patch_embeds = ref_outputs.last_hidden_state[:, 1:, :]
        
        patch_embeds = patch_embeds / patch_embeds.norm(p=2, dim=-1, keepdim=True)
        ref_patch_embeds = ref_patch_embeds / ref_patch_embeds.norm(p=2, dim=-1, keepdim=True)
        
        # Compute cosine similarity per token: (B, N, D) * (B, N, D) -> (B, N) sum(dim=-1)
        patch_sim_per_token = (patch_embeds * ref_patch_embeds).sum(dim=-1)
        
        patch_sim = patch_sim_per_token.mean(dim=-1) # (B,)
        
        # Weighted combination
        sim = alpha * cls_sim + (1 - alpha) * patch_sim
        return sim


def main():
    scorer = DinoScorer(
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    images=[
    "assets/test.jpg",
    "assets/test.jpg"
    ]
    
    try:
        pil_images = [Image.open(img).convert('RGB') for img in images]
    except Exception as e:
        print(f"Could not load images: {e}")
        return

    images = [np.array(img) for img in pil_images]
    images = np.array(images)
    images = images.transpose(0, 3, 1, 2)  # NHWC -> NCHW
    images = torch.tensor(images, dtype=torch.uint8)/255.0
    
    # Test weighted similarity
    print("Weighted Self-similarity (alpha=0.7):", scorer.image_similarity(images, images, alpha=0.7))

if __name__ == "__main__":
    main()
