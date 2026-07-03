"""
Dataset loader for LOL, ExDark, and IE datasets
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
import random


class LowLightDataset(Dataset):
    """
    Low-light image enhancement dataset
    Supports paired (low/normal) and unpaired (low only) modes
    """
    def __init__(self, root_dir, mode='train', dataset_name='LOL', 
                 img_size=(400, 600), paired=True):
        """
        Args:
            root_dir: Root directory of dataset
            mode: 'train' or 'test'
            dataset_name: 'LOL', 'ExDark', or 'IE'
            img_size: (H, W) resize target
            paired: True for paired data, False for unpaired
        """
        self.root_dir = root_dir
        self.mode = mode
        self.dataset_name = dataset_name
        self.img_size = img_size
        self.paired = paired

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        self.low_paths = []
        self.normal_paths = []

        if dataset_name == 'LOL':
            low_dir = os.path.join(root_dir, mode, 'low')
            high_dir = os.path.join(root_dir, mode, 'high')
            self.low_paths = sorted(glob.glob(os.path.join(low_dir, '*.png')))
            self.normal_paths = sorted(glob.glob(os.path.join(high_dir, '*.png')))

        elif dataset_name == 'ExDark':
            # ExDark structure: images/ class folders
            img_dir = os.path.join(root_dir, 'images')
            self.low_paths = sorted(glob.glob(os.path.join(img_dir, '**', '*.jpg'), recursive=True))

        elif dataset_name == 'IE':
            low_dir = os.path.join(root_dir, mode, 'low')
            high_dir = os.path.join(root_dir, mode, 'high')
            self.low_paths = sorted(glob.glob(os.path.join(low_dir, '*.png')))
            self.normal_paths = sorted(glob.glob(os.path.join(high_dir, '*.png')))

        if paired and len(self.normal_paths) > 0:
            assert len(self.low_paths) == len(self.normal_paths),                 f"Mismatched pairs: {len(self.low_paths)} vs {len(self.normal_paths)}"

    def __len__(self):
        return len(self.low_paths)

    def __getitem__(self, idx):
        low_path = self.low_paths[idx]
        low_img = Image.open(low_path).convert('RGB')
        low_tensor = self.transform(low_img)

        if self.paired and len(self.normal_paths) > 0:
            normal_path = self.normal_paths[idx]
            normal_img = Image.open(normal_path).convert('RGB')
            normal_tensor = self.transform(normal_img)
            return {'low': low_tensor, 'normal': normal_tensor, 'path': low_path}
        else:
            return {'low': low_tensor, 'path': low_path}


class UnpairedNormalDataset(Dataset):
    """
    Dataset for normal-light images only (used for pseudo-label generation)
    """
    def __init__(self, root_dir, img_size=(400, 600)):
        self.root_dir = root_dir
        self.img_size = img_size

        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])

        self.paths = sorted(glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True))
        self.paths += sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        return self.transform(img)


def get_dataloader(dataset_name, root_dir, mode='train', batch_size=8, 
                   img_size=(400, 600), num_workers=4, paired=True):
    """Convenience function to create dataloader"""
    dataset = LowLightDataset(root_dir, mode=mode, dataset_name=dataset_name,
                              img_size=img_size, paired=paired)

    shuffle = (mode == 'train')
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=num_workers, pin_memory=True)
    return loader


if __name__ == "__main__":
    # Test
    ds = LowLightDataset('./data/LOL', mode='train', dataset_name='LOL')
    print(f"Dataset size: {len(ds)}")
    sample = ds[0]
    print(f"Low shape: {sample['low'].shape}, Normal shape: {sample['normal'].shape}")
