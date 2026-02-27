import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_hologram(input_path, output_path=None):
    if output_path is None:
        base = input_path.rsplit(".", 1)[0] if "." in input_path else input_path
        output_path = f"{base}_hologram.png"

    img = plt.imread(input_path)
    if img.ndim == 3:
        img = img.mean(axis=2)
    img = img.astype(float)
    img = img / img.max()

    phi = img * np.pi
    E0 = np.exp(1j * phi)

    wavelength = 633e-9
    z = 0.1
    dx = 10e-6

    Nx, Ny = E0.shape
    fx = np.fft.fftfreq(Nx, dx)
    fy = np.fft.fftfreq(Ny, dx)
    FX, FY = np.meshgrid(fx, fy, indexing='ij')
    H = np.exp(-1j * np.pi * wavelength * z * (FX**2 + FY**2))

    E1 = np.fft.ifft2(np.fft.fft2(E0) * H)
    I = np.abs(E1)**2
    I = I / I.max()

    plt.imsave(output_path, I, cmap='bwr')
    return output_path


if __name__ == "__main__":
    input_path = sys.argv[1] if len(sys.argv) > 1 else "7.webp"
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    out = run_hologram(input_path, output_path)
    print(f"Saved: {out}")
