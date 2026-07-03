"""
Testing/Inference script for BioRetEcho
Evaluates PSNR, SSIM, LPIPS (AlexNet backbone) and saves results
"""
import os
import argparse
import torch
import time
from tqdm import tqdm
import cv2
import numpy as np

from model import BioRetEcho
from dataset import get_dataloader
from utils import calculate_psnr, calculate_ssim, save_image, visualize_results, compute_flops


def get_args():
    parser = argparse.ArgumentParser(description='Test BioRetEcho')
    parser.add_argument('--dataset', type=str, default='LOL', choices=['LOL', 'ExDark', 'IE'])
    parser.add_argument('--data_root', type=str, default='./data/LOL')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Output directory')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 600], help='Image size (H W)')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    parser.add_argument('--save_images', action='store_true', default=True, help='Save enhanced images')
    parser.add_argument('--save_visuals', action='store_true', default=False, help='Save comparison visualizations')
    parser.add_argument('--compute_flops', action='store_true', default=False, help='Compute FLOPs and params')
    return parser.parse_args()


def test_model(model, dataloader, device, args):
    model.eval()

    psnr_list = []
    ssim_list = []
    lpips_list = []
    time_list = []

    os.makedirs(args.output_dir, exist_ok=True)

    # Load LPIPS model (AlexNet backbone)
    try:
        import lpips
        loss_fn = lpips.LPIPS(net='alex').to(device)
        use_lpips = True
    except ImportError:
        print("Warning: lpips not installed. LPIPS will not be computed. Install: pip install lpips")
        use_lpips = False

    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc='Testing')):
            low = batch['low'].to(device)
            path = batch['path'][0] if isinstance(batch['path'], list) else batch['path']

            # Inference timing
            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_start = time.time()

            enhanced, R, S, spikes = model(low)

            torch.cuda.synchronize() if device.type == 'cuda' else None
            t_end = time.time()
            time_list.append(t_end - t_start)

            # Metrics
            if 'normal' in batch and batch['normal'] is not None:
                normal = batch['normal'].to(device)
                psnr = calculate_psnr(enhanced, normal)
                ssim = calculate_ssim(enhanced, normal)
                psnr_list.append(psnr)
                ssim_list.append(ssim)

                if use_lpips:
                    lpips_val = loss_fn(enhanced, normal).item()
                    lpips_list.append(lpips_val)

                # Save comparison visualization
                if args.save_visuals:
                    vis_path = os.path.join(args.output_dir, 'visuals', f'{idx:04d}.png')
                    visualize_results(low, enhanced, normal, save_path=vis_path)

            # Save enhanced image
            if args.save_images:
                fname = os.path.basename(path).replace('.png', '_enhanced.png').replace('.jpg', '_enhanced.png')
                save_path = os.path.join(args.output_dir, 'enhanced', fname)
                save_image(enhanced, save_path)

                # Save R and S components
                r_path = os.path.join(args.output_dir, 'components', f'R_{idx:04d}.png')
                s_path = os.path.join(args.output_dir, 'components', f'S_{idx:04d}.png')
                save_image(R, r_path)
                save_image(S, s_path)

    # Summary
    print("\n" + "="*60)
    print(f"{'Test Results':^60}")
    print("="*60)
    if psnr_list:
        print(f"PSNR:  {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB")
        print(f"SSIM:  {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}")
    if lpips_list:
        print(f"LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}")
    print(f"Time:  {np.mean(time_list)*1000:.2f} ± {np.std(time_list)*1000:.2f} ms per image")
    print(f"FPS:   {1.0/np.mean(time_list):.2f}")
    print("="*60)

    return psnr_list, ssim_list, lpips_list, time_list


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Load model
    model = BioRetEcho(T=3).to(device)

    if os.path.isfile(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}")

    # Compute FLOPs and params
    if args.compute_flops:
        flops, params = compute_flops(model, input_size=(1, 3, args.img_size[0], args.img_size[1]))
        if flops and params:
            print(f"FLOPs: {flops/1e9:.2f}G, Params: {params/1e3:.1f}K")

    # Dataloader
    test_loader = get_dataloader(args.dataset, args.data_root, mode='test',
                                  batch_size=1, img_size=tuple(args.img_size),
                                  num_workers=0, paired=True)

    # Test
    psnr_list, ssim_list, lpips_list, time_list = test_model(model, test_loader, device, args)

    # Save results to file
    result_file = os.path.join(args.output_dir, 'results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n\n")
        if psnr_list:
            f.write(f"PSNR:  {np.mean(psnr_list):.2f} ± {np.std(psnr_list):.2f} dB\n")
            f.write(f"SSIM:  {np.mean(ssim_list):.4f} ± {np.std(ssim_list):.4f}\n")
        if lpips_list:
            f.write(f"LPIPS: {np.mean(lpips_list):.4f} ± {np.std(lpips_list):.4f}\n")
        f.write(f"Time:  {np.mean(time_list)*1000:.2f} ± {np.std(time_list)*1000:.2f} ms\n")
        f.write(f"FPS:   {1.0/np.mean(time_list):.2f}\n")

    print(f"Results saved to: {result_file}")


if __name__ == "__main__":
    main()
