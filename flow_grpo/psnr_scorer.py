import torch
import numpy as np
import math


class PSNRScorer(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device

    def __call__(self, images, target_images):
        B, H, W, C = images.shape
        scores = []
        for i in range(B):
            score = calculate_psnr(images[i], target_images[i])
            scores.append(score)
        return np.array(scores)



def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_psnr_torch(img1, img2):
    img1 = img1.to(torch.float64)
    img2 = img2.to(torch.float64)
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return torch.inf
    return 20 * torch.log10(255.0 / torch.sqrt(mse))