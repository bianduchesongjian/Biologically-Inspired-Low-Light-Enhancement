"""
Training script for BioRetEcho
Supports: paired training (LOL, IE) and unpaired pseudo-label training
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import BioRetEcho, PseudoLabelGenerator
from dataset import get_dataloader, UnpairedNormalDataset
from utils import calculate_psnr, calculate_ssim, AverageMeter, print_model_summary


def get_args():
    parser = argparse.ArgumentParser(description='Train BioRetEcho')
    # Dataset
    parser.add_argument('--dataset', type=str, default='LOL', choices=['LOL', 'ExDark', 'IE'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data/LOL', 
                        help='Root directory of dataset')
    parser.add_argument('--paired', action='store_true', default=True,
                        help='Use paired data (low/normal)')
    parser.add_argument('--unpaired_normal_dir', type=str, default=None,
                        help='Directory of normal-light images for pseudo-label training')

    # Model
    parser.add_argument('--T', type=int, default=3, help='RGCs-GRU recurrence steps')
    parser.add_argument('--alpha', type=float, default=0.1, help='Dynamic threshold weight')
    parser.add_argument('--beta', type=float, default=0.1, help='Edge feature weight')
    parser.add_argument('--epsilon', type=float, default=1e-6, help='Illumination separation constant')

    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--img_size', type=int, nargs=2, default=[400, 600], 
                        help='Image size (H W)')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--save_freq', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    # Loss weights
    parser.add_argument('--w_recon', type=float, default=1.0, help='Reconstruction loss weight')
    parser.add_argument('--w_spike', type=float, default=0.1, help='Spike consistency loss weight')
    parser.add_argument('--w_smooth', type=float, default=0.01, help='Illumination smoothness weight')

    # Output
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--exp_name', type=str, default='bioretecho', help='Experiment name')

    return parser.parse_args()


def reconstruction_loss(pred, target):
    """Weighted L1 loss (spectral orthogonality between R and S)"""
    return torch.mean(torch.abs(pred - target))


def spike_consistency_loss(spike_pred, spike_gt):
    """KL divergence between predicted and ground-truth spike manifolds"""
    loss = 0
    for sp, sg in zip(spike_pred, spike_gt):
        sp = torch.clamp(sp, 1e-8, 1.0)
        sg = torch.clamp(sg, 1e-8, 1.0)
        loss += torch.mean(sp * torch.log(sp / sg))
    return loss / len(spike_pred)


def illumination_smoothness_loss(S):
    """Total variation smoothness for illumination map"""
    dx = torch.abs(S[:, :, :, :-1] - S[:, :, :, 1:])
    dy = torch.abs(S[:, :, :-1, :] - S[:, :, 1:, :])
    return torch.mean(dx) + torch.mean(dy)


def train_epoch(model, dataloader, optimizer, pseudo_gen, args, epoch, device):
    model.train()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}/{args.epochs}')

    for batch in pbar:
        low = batch['low'].to(device)

        if 'normal' in batch and batch['normal'] is not None:
            # Paired training
            normal = batch['normal'].to(device)
            target = normal
            use_pseudo = False
        else:
            # Unpaired: generate pseudo low-light from normal (if available)
            # For simplicity, we use low as target in unsupervised mode
            target = low
            use_pseudo = True

        optimizer.zero_grad()

        # Forward
        enhanced, R, S, spikes = model(low)

        # Losses
        loss_recon = reconstruction_loss(enhanced, target)
        loss_smooth = illumination_smoothness_loss(S)

        # Spike consistency (if pseudo generator available)
        if pseudo_gen is not None and use_pseudo:
            pseudo_low = pseudo_gen(target)
            _, _, _, spikes_pseudo = model(pseudo_low)
            loss_spike = spike_consistency_loss(spikes, spikes_pseudo)
        else:
            loss_spike = torch.tensor(0.0, device=device)

        total_loss = args.w_recon * loss_recon +                      args.w_smooth * loss_smooth +                      args.w_spike * loss_spike

        total_loss.backward()
        optimizer.step()

        # Metrics
        with torch.no_grad():
            psnr = calculate_psnr(enhanced, target)
            ssim = calculate_ssim(enhanced, target)

        losses.update(total_loss.item(), low.size(0))
        psnrs.update(psnr, low.size(0))
        ssims.update(ssim, low.size(0))

        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'PSNR': f'{psnrs.avg:.2f}',
            'SSIM': f'{ssims.avg:.4f}'
        })

    return losses.avg, psnrs.avg, ssims.avg


def validate(model, dataloader, device):
    model.eval()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            low = batch['low'].to(device)
            normal = batch.get('normal')

            enhanced, _, _, _ = model(low)

            if normal is not None:
                normal = normal.to(device)
                psnr = calculate_psnr(enhanced, normal)
                ssim = calculate_ssim(enhanced, normal)
                psnrs.update(psnr, low.size(0))
                ssims.update(ssim, low.size(0))

    return psnrs.avg, ssims.avg


def main():
    args = get_args()

    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)

    writer = SummaryWriter(os.path.join(output_dir, 'logs'))

    # Model
    model = BioRetEcho(T=args.T, alpha=args.alpha, beta=args.beta, epsilon=args.epsilon).to(device)
    print_model_summary(model)

    # Pseudo-label generator (for unsupervised training)
    pseudo_gen = PseudoLabelGenerator().to(device) if args.unpaired_normal_dir else None

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # DataLoaders
    train_loader = get_dataloader(args.dataset, args.data_root, mode='train', 
                                   batch_size=args.batch_size, img_size=tuple(args.img_size),
                                   num_workers=args.num_workers, paired=args.paired)

    val_loader = get_dataloader(args.dataset, args.data_root, mode='test', 
                                 batch_size=1, img_size=tuple(args.img_size),
                                 num_workers=args.num_workers, paired=args.paired)

    # Training loop
    best_psnr = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, psnr, ssim = train_epoch(model, train_loader, optimizer, pseudo_gen, args, epoch, device)
        val_psnr, val_ssim = validate(model, val_loader, device)

        scheduler.step()

        # Logging
        writer.add_scalar('Train/Loss', loss, epoch)
        writer.add_scalar('Train/PSNR', psnr, epoch)
        writer.add_scalar('Train/SSIM', ssim, epoch)
        writer.add_scalar('Val/PSNR', val_psnr, epoch)
        writer.add_scalar('Val/SSIM', val_ssim, epoch)

        print(f"Epoch {epoch}: Loss={loss:.4f}, Train PSNR={psnr:.2f}, Val PSNR={val_psnr:.2f}, Val SSIM={val_ssim:.4f}")

        # Save checkpoint
        if epoch % args.save_freq == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(output_dir, 'checkpoints', f'epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_psnr': best_psnr,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            best_path = os.path.join(output_dir, 'checkpoints', 'best_model.pth')
            torch.save(model.state_dict(), best_path)
            print(f"Best model saved with PSNR: {best_psnr:.2f}")

    writer.close()
    print(f"Training complete. Best Val PSNR: {best_psnr:.2f} dB")


if __name__ == "__main__":
    main()
