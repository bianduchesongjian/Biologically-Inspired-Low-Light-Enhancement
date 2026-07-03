"""
Utility functions for training, evaluation, and visualization
"""
import torch
import torch.nn.functional as F
import numpy as np
from math import log10
import os
import cv2


def calculate_psnr(img1, img2, max_val=1.0):
    """
    Calculate PSNR between two images
    Args:
        img1, img2: Tensors of shape (B, C, H, W), range [0, 1]
    Returns:
        PSNR value in dB
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * log10(max_val / torch.sqrt(mse).item())


def calculate_ssim(img1, img2, window_size=11, size_average=True):
    """
    Calculate SSIM between two images
    Args:
        img1, img2: Tensors of shape (B, C, H, W), range [0, 1]
    Returns:
        SSIM value
    """
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, window_size, 1, padding=window_size//2)
    mu2 = F.avg_pool2d(img2, window_size, 1, padding=window_size//2)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 ** 2, window_size, 1, padding=window_size//2) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 ** 2, window_size, 1, padding=window_size//2) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size, 1, padding=window_size//2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) /                ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def save_image(tensor, path):
    """Save a tensor as an image file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img = tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def visualize_results(low, enhanced, normal=None, save_path=None):
    """
    Visualize comparison: low-light, enhanced, (optional) normal
    """
    import matplotlib.pyplot as plt

    if normal is not None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(low[0].cpu().permute(1, 2, 0).numpy())
        axes[0].set_title('Low-Light Input')
        axes[1].imshow(enhanced[0].cpu().permute(1, 2, 0).numpy())
        axes[1].set_title('Enhanced (BioRetEcho)')
        axes[2].imshow(normal[0].cpu().permute(1, 2, 0).numpy())
        axes[2].set_title('Ground Truth')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(low[0].cpu().permute(1, 2, 0).numpy())
        axes[0].set_title('Low-Light Input')
        axes[1].imshow(enhanced[0].cpu().permute(1, 2, 0).numpy())
        axes[1].set_title('Enhanced (BioRetEcho)')

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_flops(model, input_size=(1, 3, 400, 600)):
    """Estimate FLOPs using thop (optional)"""
    try:
        from thop import profile
        dummy_input = torch.randn(input_size)
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        return flops, params
    except ImportError:
        print("thop not installed. Install via: pip install thop")
        return None, None


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model):
    """Print model architecture summary"""
    total, trainable = count_parameters(model)
    print("=" * 60)
    print(f"{'Model Summary':^60}")
    print("=" * 60)
    print(f"Total parameters:      {total:,} ({total/1e3:.1f}K)")
    print(f"Trainable parameters:  {trainable:,} ({trainable/1e3:.1f}K)")
    print("=" * 60)
