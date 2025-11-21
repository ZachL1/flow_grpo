import torch
import torch.nn as nn
import lpips


class LPIPSScorer(torch.nn.Module):
    """
    LPIPS (Learned Perceptual Image Patch Similarity) scorer.
    Computes perceptual similarity between two images.
    Lower LPIPS values indicate more similar images.
    We return negative LPIPS so that higher scores mean better similarity.
    """
    def __init__(self, net='alex', device='cuda'):
        """
        Args:
            net: Network to use for feature extraction. Options: 'alex', 'vgg', 'squeeze'
            device: Device to run the model on
        """
        super().__init__()
        self.device = device
        self.loss_fn = lpips.LPIPS(net=net).to(device)
        self.loss_fn.eval()

    @torch.no_grad()
    def __call__(self, images, target_images):
        """
        Compute LPIPS score between images and target images.
        
        Args:
            images: Tensor of shape (B, C, H, W) with values in [0, 1]
            target_images: Tensor of shape (B, C, H, W) with values in [0, 1]
        
        Returns:
            scores: Negative LPIPS distances (higher is better)
        """
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        if not isinstance(target_images, torch.Tensor):
            target_images = torch.tensor(target_images)
        
        images = images.to(self.device)
        target_images = target_images.to(self.device)
        
        # LPIPS expects images in range [-1, 1]
        images = images * 2 - 1
        target_images = target_images * 2 - 1
        
        # Compute LPIPS distance
        distances = self.loss_fn(images, target_images)
        
        # Return negative distance so higher scores mean better similarity
        # Squeeze to remove extra dimensions
        scores = 1 - distances.squeeze((1,2,3))
        
        return scores


def main():
    import torchvision.transforms as transforms
    from PIL import Image
    import numpy as np

    # Test the scorer
    scorer = LPIPSScorer(device='cuda' if torch.cuda.is_available() else 'cpu')

    # Create dummy images for testing
    image1 = torch.rand(2, 3, 256, 256)
    image2 = torch.rand(2, 3, 256, 256)
    
    scores = scorer(image1, image2)
    print("LPIPS scores:", scores)
    print("Scores shape:", scores.shape)


if __name__ == "__main__":
    main()

