# -*- coding: utf-8 -*-
"""
QIUP single-image transformation (Nature 2014 paper style).
Takes one image, applies the QIUP simulation, and saves the result figure.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')


class QIUPTransformer:
    """
    QIUP transformation for a single image. Outputs: target, T, gamma,
    holo_constructive, holo_destructive, holo_sum, holo_diff.
    """

    def __init__(self,
                 img_size=512,
                 output_size=256,
                 visibility=0.77,
                 mean_photons=300,
                 qe=0.95,
                 em_gain=800,
                 cic_rate=0.002,
                 read_noise_sigma=0.5,
                 dark_current=0.0,
                 preamp_gain=2.0,
                 phase_stability=0.9,
                 background_threshold=0.12,  # آستانهٔ شدت: پیکسل‌های ضعیف‌تر از این مقدار مشکی (صفر) می‌شوند؛ عدد بزرگ‌تر = پس‌زمینه مشکی‌تر
                 signal_threshold=0.05,
                 add_shot_noise=True,
                 add_cic=True,
                 add_read_noise=True,
                 add_phase_disturbance=True):
        self.img_size = img_size
        self.output_size = output_size
        self.visibility = visibility
        self.mean_photons = mean_photons
        self.qe = qe
        self.em_gain = em_gain
        self.cic_rate = cic_rate
        self.read_noise_sigma = read_noise_sigma
        self.dark_current = dark_current
        self.preamp_gain = preamp_gain
        self.phase_stability = phase_stability
        self.background_threshold = background_threshold
        self.signal_threshold = signal_threshold
        self.add_shot_noise = add_shot_noise
        self.add_cic = add_cic
        self.add_read_noise = add_read_noise
        self.add_phase_disturbance = add_phase_disturbance

    def prepare_object_properties(self, image):
        img_norm = (image - image.min()) / (image.max() - image.min() + 1e-8)
        T = img_norm * 0.8 + 0.2
        T = np.clip(T, 0, 1)
        gamma = 2 * np.pi * (img_norm - 0.5)
        return T, gamma

    def generate_phase_disturbance(self, shape):
        h, w = shape
        x = np.linspace(0, 2*np.pi, w)
        y = np.linspace(0, 2*np.pi, h)
        X, Y = np.meshgrid(x, y)
        disturbance = (0.02 * np.sin(X) + 0.02 * np.sin(Y))
        random_phase = np.random.randn(h, w) * 0.03
        random_phase = gaussian_filter(random_phase, sigma=4)
        return (disturbance + random_phase) * (1 - self.phase_stability) * 0.3

    def apply_nature_style_threshold(self, hologram, T):
        processed = hologram.copy()
        signal_mask = T > self.signal_threshold
        processed[processed < self.background_threshold] = 0
        processed[~signal_mask] = 0
        if signal_mask.any():
            min_val = processed[signal_mask].min()
            max_val = processed[signal_mask].max()
            if max_val > min_val:
                processed[signal_mask] = (processed[signal_mask] - min_val) / (max_val - min_val)
        return processed

    def apply_emccd_noise(self, intensity, T):
        photons = intensity * self.mean_photons
        electrons = photons * self.qe
        if self.add_shot_noise:
            electrons = np.random.poisson(electrons)
        if self.add_cic:
            cic = np.random.poisson(self.cic_rate, size=electrons.shape)
            signal_mask = T > self.signal_threshold
            electrons = electrons + cic * signal_mask
        if self.em_gain > 1:
            excess_noise = np.random.gamma(2, 0.5, electrons.shape)
            effective_gain = self.em_gain * (0.8 + 0.2 * excess_noise)
            electrons = electrons * effective_gain
        if self.add_read_noise:
            electrons = electrons + np.random.randn(*electrons.shape) * self.read_noise_sigma
        electrons = np.maximum(electrons, 0)
        adu = np.clip(electrons / self.preamp_gain, 0, 65535)
        hologram = adu.astype(np.float32) / 65535.0
        return self.apply_nature_style_threshold(hologram, T)

    def simulate_three_outputs(self, T, gamma):
        if self.add_phase_disturbance:
            total_phase = gamma + self.generate_phase_disturbance(T.shape)
        else:
            total_phase = gamma
        I_c = 0.5 * (1 + self.visibility * T * np.cos(total_phase))
        I_d = 0.5 * (1 + self.visibility * T * np.cos(total_phase + np.pi))
        holo_c = self.apply_emccd_noise(I_c, T)
        holo_d = self.apply_emccd_noise(I_d, T)
        holo_sum = self.apply_nature_style_threshold((holo_c + holo_d) / 2, T)
        holo_diff = self.apply_nature_style_threshold((holo_c - holo_d) / 2 + 0.5, T)
        return holo_c, holo_d, holo_sum, holo_diff

    def _resize(self, img):
        return np.array(Image.fromarray(
            (img * 65535).astype(np.uint16)).resize(
            (self.img_size, self.img_size), Image.LANCZOS)) / 65535.0

    def transform(self, image):
        """Apply QIUP transformation. image: numpy grayscale [0,1] or uint8."""
        if image.dtype == np.uint8 or image.max() > 1.0:
            image = image.astype(np.float32) / 255.0
        if len(image.shape) == 3:
            image = np.mean(image, axis=2)
        img_np = np.array(Image.fromarray(
            (image * 65535).clip(0, 65535).astype(np.uint16)).resize(
            (self.output_size, self.output_size), Image.LANCZOS)) / 65535.0

        T, gamma = self.prepare_object_properties(img_np)
        holo_c, holo_d, holo_sum, holo_diff = self.simulate_three_outputs(T, gamma)

        def resized(arr):
            return np.array(Image.fromarray(
                (np.clip(arr, 0, 1) * 65535).astype(np.uint16)).resize(
                (self.output_size, self.output_size), Image.LANCZOS)) / 65535.0

        return {
            'target': resized(img_np),
            'T': resized(T),
            'gamma': resized((gamma + np.pi) / (2 * np.pi)),
            'holo_constructive': self._resize(holo_c),
            'holo_destructive': self._resize(holo_d),
            'holo_sum': self._resize(holo_sum),
            'holo_diff': self._resize(holo_diff),
        }


def load_image(path):
    """Load image as grayscale numpy array in [0, 1]."""
    img = Image.open(path).convert('L')
    return np.array(img, dtype=np.float32) / 255.0


def result_to_figure(result):
    """Build the Nature-style 2x4 result figure (no display)."""
    cmap = plt.get_cmap('Blues_r')

    def to_rgb(arr):
        c = cmap(np.asarray(arr).squeeze())
        return (c[:, :, :3] * 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('QIUP transformation (Nature figure 3 style)', fontsize=16)

    axes[0, 0].imshow(result['target'], cmap='gray')
    axes[0, 0].set_title('a) Ground Truth')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(result['T'], cmap='gray', vmin=0, vmax=1)
    axes[0, 1].set_title('b) Transmission T')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(result['gamma'], cmap='twilight', vmin=0, vmax=1)
    axes[0, 2].set_title('c) Phase shift γ')
    axes[0, 2].axis('off')

    axes[0, 3].axis('off')

    axes[1, 0].imshow(to_rgb(result['holo_constructive']))
    axes[1, 0].set_title('d) Constructive output')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(to_rgb(result['holo_destructive']))
    axes[1, 1].set_title('e) Destructive output')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(to_rgb(result['holo_sum']))
    axes[1, 2].set_title('f) Sum of two outputs')
    axes[1, 2].axis('off')

    axes[1, 3].imshow(to_rgb(result['holo_diff']))
    axes[1, 3].set_title('g) Difference of two outputs')
    axes[1, 3].axis('off')

    plt.tight_layout()
    return fig


def transform_and_save_figure(image_path, output_path=None, **transformer_kw):
    """
    Load one image, run QIUP transformation, and save the result figure.
    If output_path is None, save to <image_stem>_result.png next to the input.
    Any transformer_kw (e.g. background_threshold, visibility) are passed to
    QIUPTransformer; otherwise the class __init__ defaults are used.
    """
    image = load_image(image_path)
    transformer = QIUPTransformer(**transformer_kw)
    result = transformer.transform(image)
    if output_path is None:
        base, _ = os.path.splitext(image_path)
        output_path = base + '_result.png'
    fig = result_to_figure(result)
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Result figure saved: {output_path}")
    return result
;j;lkj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QIUP single-image transformation; save result figure.")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output figure path (default: <image>_result.png)")
    args = parser.parse_args()

    transform_and_save_figure(args.image, output_path=args.output)
