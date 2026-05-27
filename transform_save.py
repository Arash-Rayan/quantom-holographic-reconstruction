import os
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class QIUPImageSimulator:

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
                 exposure_time=1.0,
                 preamp_gain=2.0,
                 phase_stability=0.9,
                 background_threshold=0.05,
                 add_shot_noise=True,
                 add_cic=True,
                 add_read_noise=True,
                 add_phase_disturbance=True):

        self.img_size              = img_size
        self.output_size           = output_size
        self.visibility            = visibility
        self.mean_photons          = mean_photons
        self.qe                    = qe
        self.em_gain               = em_gain
        self.cic_rate              = cic_rate
        self.read_noise_sigma      = read_noise_sigma
        self.dark_current          = dark_current
        self.exposure_time         = exposure_time
        self.preamp_gain           = preamp_gain
        self.phase_stability       = phase_stability
        self.background_threshold  = background_threshold
        self.add_shot_noise        = add_shot_noise
        self.add_cic               = add_cic
        self.add_read_noise        = add_read_noise
        self.add_phase_disturbance = add_phase_disturbance

        print("=" * 60)
        print("QIUP IMAGE SIMULATOR")
        print("=" * 60)
        print(f"Equation : P = 0.5 * (1 + {visibility} * T * cos(gamma))")
        print(f"EM Gain  : {em_gain}")
        print(f"QE       : {qe}")
        print("=" * 60)

    # =====================================================
    # OBJECT PREPARATION
    # =====================================================

    def _prepare_object(self, image):

        img_norm = (
            image - image.min()
        ) / (
            image.max() - image.min() + 1e-8
        )

        T = img_norm * 0.8 + 0.2

        T[T < self.background_threshold] = 0

        T = np.clip(T, 0, 1)

        gamma = 2 * np.pi * (img_norm - 0.5)

        return T, gamma

    # =====================================================
    # PHASE DISTURBANCE
    # =====================================================

    def _phase_disturbance(self, shape):

        h, w = shape

        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)

        X, Y = np.meshgrid(x, y)

        dist = (
            0.05 * np.sin(X)
            + 0.05 * np.sin(Y)
            + 0.03 * np.sin(2 * X + Y)
            + 0.02 * np.sin(3 * X - 2 * Y)
        )

        rnd = gaussian_filter(
            np.random.randn(h, w) * 0.1,
            sigma=3
        )

        return (dist + rnd) * (1 - self.phase_stability)

    # =====================================================
    # HOLOGRAM SIMULATION
    # =====================================================

    def simulate_hologram(self, image):

        T, gamma = self._prepare_object(image)

        total_phase = gamma + (
            self._phase_disturbance(image.shape)
            if self.add_phase_disturbance else 0
        )

        photons = (
            0.5
            * (
                1
                + self.visibility
                * T
                * np.cos(total_phase)
            )
            * self.mean_photons
        )

        electrons = photons * self.qe

        # -------------------------------------------------

        if self.add_shot_noise:

            electrons = np.random.poisson(
                electrons
            ).astype(np.float64)

        # -------------------------------------------------

        if self.add_cic:

            electrons += np.random.poisson(
                self.cic_rate,
                size=electrons.shape
            )

        # -------------------------------------------------

        if self.em_gain > 1:

            excess = np.random.gamma(
                2,
                0.5,
                electrons.shape
            )

            electrons *= self.em_gain * (
                0.8 + 0.2 * excess
            )

        # -------------------------------------------------

        if self.add_read_noise:

            electrons += (
                np.random.randn(*electrons.shape)
                * self.read_noise_sigma
            )

        # -------------------------------------------------

        electrons = np.maximum(electrons, 0)

        adu = np.clip(
            electrons / self.preamp_gain,
            0,
            65535
        )

        hologram = (
            np.round(adu)
            .astype(np.uint16)
            .astype(np.float32)
            / 65535.0
        )

        hologram[
            hologram < self.background_threshold
        ] = 0

        return hologram, T, gamma

    # =====================================================
    # SAVE USING SAME VISUALIZATION AS MATPLOTLIB
    # =====================================================

    def save_visualized(self,
                        image,
                        path,
                        cmap='gray'):

        plt.imsave(
            path,
            image,
            cmap=cmap
        )

    # =====================================================
    # MAIN FUNCTION
    # =====================================================

    def process_image(self,
                      image_path,
                      save_dir='output',
                      show=True):

        os.makedirs(save_dir, exist_ok=True)

        # -------------------------------------------------
        # LOAD IMAGE
        # -------------------------------------------------

        img = Image.open(image_path).convert('L')

        img = img.resize(
            (self.output_size, self.output_size),
            Image.LANCZOS
        )

        img_np = (
            np.array(img).astype(np.float32)
            / 255.0
        )

        # -------------------------------------------------
        # SIMULATE
        # -------------------------------------------------

        hologram, T, gamma = self.simulate_hologram(img_np)

        # -------------------------------------------------
        # RESIZE HOLOGRAM
        # -------------------------------------------------

        holo_resized = np.array(
            Image.fromarray(
                (hologram * 65535).astype(np.uint16)
            ).resize(
                (self.img_size, self.img_size),
                Image.LANCZOS
            )
        ) / 65535.0

        # -------------------------------------------------
        # TARGET
        # -------------------------------------------------

        target = np.array(
            Image.fromarray(
                (img_np * 255).astype(np.uint8)
            ).resize(
                (self.output_size, self.output_size),
                Image.NEAREST
            )
        ) / 255.0

        target = (
            target - target.min()
        ) / (
            target.max() - target.min() + 1e-8
        )

        # =================================================
        # SAVE EVERYTHING
        # =================================================

        self.save_visualized(
            target,
            os.path.join(save_dir, 'target.png'),
            cmap='gray'
        )

        self.save_visualized(
            holo_resized,
            os.path.join(save_dir, 'hologram.png'),
            cmap='gray'
        )

        self.save_visualized(
            T,
            os.path.join(save_dir, 'T.png'),
            cmap='gray'
        )

        self.save_visualized(
            gamma,
            os.path.join(save_dir, 'gamma.png'),
            cmap='jet'
        )

        print(f"\nSaved outputs to: {save_dir}")

        # =================================================
        # SHOW
        # =================================================

        if show:

            fig, ax = plt.subplots(
                1,
                4,
                figsize=(18, 5)
            )

            ax[0].imshow(
                target,
                cmap='gray'
            )
            ax[0].set_title('Target')

            ax[1].imshow(
                holo_resized,
                cmap='gray'
            )
            ax[1].set_title('Hologram')

            ax[2].imshow(
                T,
                cmap='gray'
            )
            ax[2].set_title('T')

            ax[3].imshow(
                gamma,
                cmap='jet'
            )
            ax[3].set_title('Gamma')

            for a in ax:
                a.axis('off')

            plt.tight_layout()
            plt.show()

        return {
            'target': target,
            'hologram': holo_resized,
            'T': T,
            'gamma': gamma
        }


# =========================================================
# USAGE
# =========================================================

simulator = QIUPImageSimulator()

results = simulator.process_image(
    image_path='solid5.png',
    save_dir='simulated_output',
    show=True
)