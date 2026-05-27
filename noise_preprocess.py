import math
import random
import torch

def simulate_qiup_hologram(
    amplitude,
    rp=0.5,
    ri=0.5,
    gamma=0.5,
    noise_std_list=(0.01, 0.05, 0.1, 0.5),
):


    phi = torch.rand_like(amplitude) * (2 * math.pi)
    rs = amplitude

    base_holo = rp**2 + rs**2 + 2 * rp * rs * ri * gamma * torch.cos(phi)

    picked_std = random.choice(noise_std_list)
    created_noise = picked_std * torch.randn_like(base_holo)

    noisy_holo = base_holo + created_noise
    noisy_holo = (noisy_holo - noisy_holo.min()) / (noisy_holo.max() - noisy_holo.min() + 1e-8)

    return noisy_holo