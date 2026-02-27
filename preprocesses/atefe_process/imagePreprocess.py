"""QHUP holographic interference pattern simulation from images."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image


class QHUPSimulatedDataset(Dataset):
    def __init__(self, 
                 dataset_name='mnist',
                 root_dir='./data',
                 img_size=512,
                 visibility=0.7,
                 add_noise=True,
                 noise_level=0.1,
                 add_phase_disturbance=True):
        self.img_size = img_size
        self.visibility = visibility
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.add_phase_disturbance = add_phase_disturbance
        gt_size = img_size // 2
        transform = transforms.Compose([
            transforms.Resize((gt_size, gt_size)),
            transforms.ToTensor()
        ])
        
        if dataset_name.lower() == 'mnist':
            self.source_dataset = datasets.MNIST(
                root=root_dir, 
                train=True, 
                download=True,
                transform=transform
            )
        elif dataset_name.lower() == 'fashion':
            self.source_dataset = datasets.FashionMNIST(
                root=root_dir, 
                train=True, 
                download=True,
                transform=transform
            )
        elif dataset_name.lower() == 'cifar':
            self.source_dataset = datasets.CIFAR10(
                root=root_dir, 
                train=True, 
                download=True,
                transform=transforms.Compose([
                    transforms.Resize((gt_size, gt_size)),
                    transforms.Grayscale(num_output_channels=1),
                    transforms.ToTensor()
                ])
            )
        else:
            raise ValueError("dataset_name must be 'mnist', 'fashion', or 'cifar'")
        
        print(f"Dataset loaded: {len(self.source_dataset)} samples")
    
    def generate_hologram(self, ground_truth_image):
        if isinstance(ground_truth_image, torch.Tensor):
            gt_np = ground_truth_image.squeeze().cpu().numpy()
        else:
            gt_np = np.asarray(ground_truth_image)
        gt_np = (gt_np - gt_np.min()) / (gt_np.max() - gt_np.min() + 1e-8)
        T = gt_np
        phase = 2 * np.pi * (gt_np - 0.5)
        if self.add_phase_disturbance:
            h, w = gt_np.shape
            x = np.linspace(0, 2*np.pi, h)
            y = np.linspace(0, 2*np.pi, w)
            X, Y = np.meshgrid(x, y)
            phase_disturbance = (
                0.2 * np.sin(X) + 
                0.2 * np.sin(2*Y) + 
                0.1 * np.sin(3*X + 2*Y) +
                0.1 * np.random.randn(h, w) * 0.1
            )
            phase = phase + phase_disturbance
        hologram = 0.5 * (1 + self.visibility * T * np.cos(phase))
        if self.add_noise:
            shot_noise = np.random.poisson(hologram * 100) / 100
            hologram = 0.8 * hologram + 0.2 * shot_noise
            gaussian_noise = np.random.randn(*hologram.shape) * self.noise_level
            hologram = hologram + gaussian_noise
            hologram = np.clip(hologram, 0, 1)
        hologram_resized = np.array(
            Image.fromarray((hologram * 255).astype(np.uint8))
            .resize((self.img_size, self.img_size), Image.BICUBIC)
        ) / 255.0
        target_size = self.img_size // 2
        gt_resized = np.array(
            Image.fromarray((gt_np * 255).astype(np.uint8))
            .resize((target_size, target_size), Image.BICUBIC)
        ) / 255.0
        
        return hologram_resized, gt_resized
    
    def __len__(self):
        return len(self.source_dataset)
    
    def __getitem__(self, idx):
        img, label = self.source_dataset[idx]
        if isinstance(img, torch.Tensor):
            img_np = img.squeeze().numpy()
        else:
            img_np = np.array(img)
        hologram, ground_truth = self.generate_hologram(img_np)
        hologram_tensor = torch.FloatTensor(hologram).unsqueeze(0)
        ground_truth_tensor = torch.FloatTensor(ground_truth).unsqueeze(0)
        return hologram_tensor, ground_truth_tensor, label


def save_image(image_array, filepath):
    img = Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8), mode='L')
    img.save(filepath)
    print(f"Saved: {filepath}")


if __name__ == "__main__":
    OUTPUT_DIR = "processed_images"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 60)
    print("QHUP Image Preprocessing - Hologram Generation")
    print("=" * 60)
    dataset = QHUPSimulatedDataset(
        dataset_name='mnist',
        img_size=512,
        visibility=0.7,
        add_noise=True,
        noise_level=0.1,
        add_phase_disturbance=True,
    )
    print("\nProcessing MNIST digits 1-9...")
    digit_samples = {}
    for hologram, ground_truth, label in dataset:
        d = int(label)
        if 1 <= d <= 9 and d not in digit_samples:
            digit_samples[d] = (hologram, ground_truth)
        if len(digit_samples) == 9:
            break
    
    print(f"Collected digits: {sorted(digit_samples.keys())}")
    for digit, (hologram, ground_truth) in digit_samples.items():
        holo_path = os.path.join(OUTPUT_DIR, f"digit_{digit}_hologram.png")
        gt_path = os.path.join(OUTPUT_DIR, f"digit_{digit}_ground_truth.png")
        
        save_image(hologram.squeeze().numpy(), holo_path)
        save_image(ground_truth.squeeze().numpy(), gt_path)
    logo_path = "cqst_logo.jpg"
    if os.path.exists(logo_path):
        print(f"\nProcessing custom image: {logo_path}...")
        logo_img = Image.open(logo_path).convert("L")
        gt_size = dataset.img_size // 2
        logo_resized = logo_img.resize((gt_size, gt_size), Image.LANCZOS)
        logo_np = np.array(logo_resized) / 255.0
        logo_hologram_np, logo_gt_np = dataset.generate_hologram(logo_np)
        logo_holo_path = os.path.join(OUTPUT_DIR, "logo_hologram.png")
        logo_gt_path = os.path.join(OUTPUT_DIR, "logo_ground_truth.png")
        
        save_image(logo_hologram_np, logo_holo_path)
        save_image(logo_gt_np, logo_gt_path)
    else:
        print(f"\nLogo file '{logo_path}' not found. Skipping custom image processing.")
    
    print(f"\n{'=' * 60}")
    print(f"All processed images saved to: {OUTPUT_DIR}/")
    print("=" * 60)
