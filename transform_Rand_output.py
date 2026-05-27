import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def transform_image(self, image):
    if isinstance(image, Image.Image):
        image = image.convert('L')
        image = image.resize((self.output_size, self.output_size), Image.LANCZOS)
        img_np = np.array(image) / 255.0
    else:
        img_np = image
        if img_np.max() > 1:
            img_np = img_np / 255.0

    img_np = np.clip(img_np, 0, 1)

    def hologram_transform(x):
        visibility = np.random.uniform(0.5, 1.0)
        mean_photons = np.random.uniform(200, 600)
        em_gain = np.random.choice([1, 200, 400, 800])
        read_noise = np.random.uniform(0.2, 1.5)

        T = x * 0.8 + 0.2
        gamma = 2 * np.pi * (x - 0.5)

        h, w = x.shape
        X, Y = np.meshgrid(
            np.linspace(0, 4*np.pi, w),
            np.linspace(0, 4*np.pi, h)
        )

        phase_noise = (
            0.05 * np.sin(X) +
            0.05 * np.sin(Y) +
            0.02 * np.sin(2*X + Y)
        )

        total_phase = gamma + phase_noise

        photons = 0.5 * (1 + visibility * T * np.cos(total_phase)) * mean_photons
        electrons = photons

        electrons = np.random.poisson(np.maximum(electrons, 0))

        if em_gain > 1:
            electrons = electrons * em_gain

        electrons += np.random.randn(*x.shape) * read_noise

        electrons = np.clip(electrons, 0, None)

        out = electrons / (electrons.max() + 1e-8)
        return out

    def noise_transform(x):
        noise_level = np.random.uniform(0.05, 0.25)
        noise = np.random.randn(*x.shape) * noise_level
        return np.clip(x + noise, 0, 1)

    def blur_transform(x):
        sigma = np.random.uniform(0.5, 2.5)
        return gaussian_filter(x, sigma=sigma)

    def contrast_transform(x):
        alpha = np.random.uniform(0.5, 1.8)
        beta = np.random.uniform(-0.2, 0.2)
        return np.clip(alpha * x + beta, 0, 1)

    def gamma_transform(x):
        gamma = np.random.uniform(0.5, 2.0)
        return np.clip(np.power(x, gamma), 0, 1)

    transforms = [
        hologram_transform,
        noise_transform,
        blur_transform,
        contrast_transform,
        gamma_transform
    ]

    transform_fn = np.random.choice(transforms)

    out = transform_fn(img_np).astype(np.float32)

    return out