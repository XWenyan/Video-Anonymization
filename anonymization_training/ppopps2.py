"""
PP-OPPS (ICCV2021)
Learning Privacy-Preserving Optics for Human Pose Estimation

Part 1
---------------------------------------
Basic Mathematics
    - Imports
    - Noll Index
    - Zernike Basis
    - Wavefront
"""

import os
import cv2
import numpy as np

from tqdm import tqdm
from scipy.special import factorial
from scipy.signal import fftconvolve


# ============================================================
#                       ZERNIKE BASIS
# ============================================================

class ZernikeBasis:
    """
    Generate Zernike Polynomials using Noll indexing.

    Supports first 37 modes used in PP-OPPS.
    """

    def __init__(self):

        # Noll indexing
        self.noll_table = [

            (0, 0),

            (1, 1),
            (1, -1),

            (2, 0),
            (2, -2),
            (2, 2),

            (3, -1),
            (3, 1),
            (3, -3),
            (3, 3),

            (4, 0),
            (4, 2),
            (4, -2),
            (4, 4),
            (4, -4),

            (5, 1),
            (5, -1),
            (5, 3),
            (5, -3),
            (5, 5),
            (5, -5),

            (6, 0),
            (6, -2),
            (6, 2),
            (6, -4),
            (6, 4),
            (6, -6),
            (6, 6),

            (7, -1),
            (7, 1),
            (7, -3),
            (7, 3),
            (7, -5),
            (7, 5),
            (7, -7),
            (7, 7)
        ]

    # --------------------------------------------------------

    @staticmethod
    def radial(n, m, rho):
        """
        Radial Zernike polynomial.

        Parameters
        ----------
        n,m : Zernike order

        rho : normalized radius
        """

        m = abs(m)

        if (n - m) % 2 != 0:
            return np.zeros_like(rho)

        radial = np.zeros_like(rho)

        for k in range((n - m) // 2 + 1):

            c = (-1) ** k

            c *= factorial(n - k)

            c /= (
                factorial(k)
                * factorial((n + m) // 2 - k)
                * factorial((n - m) // 2 - k)
            )

            radial += c * rho ** (n - 2 * k)

        return radial

    # --------------------------------------------------------

    def polynomial(self, n, m, rho, theta):

        R = self.radial(n, m, rho)

        if m == 0:
            norm = np.sqrt(n + 1.0)
        else:
            norm = np.sqrt(2.0 * (n + 1.0))

        if m > 0:
            return norm * R * np.cos(m * theta)

        elif m < 0:
            return norm * R * np.sin(abs(m) * theta)

        else:
            return norm * R

    # --------------------------------------------------------

    def from_noll(self, index, rho, theta):
        """
        Compute Zernike basis by Noll index.

        index starts from 1.
        """

        if index < 1 or index > len(self.noll_table):
            raise ValueError("Unsupported Noll index")

        n, m = self.noll_table[index - 1]

        return self.polynomial(n, m, rho, theta)


# ============================================================
#                     WAVEFRONT MODEL
# ============================================================

class PhaseAberration:
    """
    Wavefront Aberration

    W(rho,theta)=Σ ai Zi
    """

    def __init__(self, coeffs):

        self.coeffs = np.asarray(coeffs, dtype=np.float64)

        self.zernike = ZernikeBasis()

    # --------------------------------------------------------

    def generate(self, rho, theta):
        """
        Generate wavefront.

        Parameters
        ----------
        rho :
            normalized radius

        theta :
            angle

        Returns
        -------
        wavefront
        """

        phase = np.zeros_like(rho, dtype=np.float64)

        num_modes = min(
            len(self.coeffs),
            len(self.zernike.noll_table)
        )

        for i in range(num_modes):

            ai = self.coeffs[i]

            if abs(ai) < 1e-12:
                continue

            Zi = self.zernike.from_noll(
                i + 1,
                rho,
                theta
            )

            phase += ai * Zi

        return phase


# ============================================================
#                DEFAULT PAPER PRESETS
# ============================================================

def paper_coefficients(mode="paper"):
    """
    Default coefficient presets.

    NOTE
    ----
    ICCV2021 does NOT release trained coefficients.

    Therefore we use physically meaningful presets that
    approximate the optical aberrations described in the paper.
    """

    coeffs = np.zeros(37, dtype=np.float64)

    if mode == "paper":

        # Defocus
        coeffs[4] = -2.0

        # Astigmatism
        coeffs[3] = 0.8
        coeffs[5] = 0.8

        # Coma
        coeffs[7] = 0.6
        coeffs[8] = 0.6

        # Spherical
        coeffs[12] = 0.25

    elif mode == "defocus":

        coeffs[4] = -3.0

    elif mode == "coma":

        coeffs[7] = 1.2
        coeffs[8] = 1.2

    elif mode == "astigmatism":

        coeffs[3] = 1.0
        coeffs[5] = 1.0

    elif mode == "weak":

        coeffs[4] = -1.2
        coeffs[7] = 0.3
        coeffs[8] = 0.3

    elif mode == "strong":

        coeffs[4] = -4.0
        coeffs[3] = 1.2
        coeffs[5] = 1.2
        coeffs[7] = 1.0
        coeffs[8] = 1.0
        coeffs[12] = 0.5

    else:
        raise ValueError("Unknown preset.")

    return coeffs
# ============================================================
#                   OPTICAL SYSTEM
# ============================================================

class OpticalSystem:
    """
    PP-OPPS Optical Encoder

    Wavefront
        ↓
    Complex Pupil
        ↓
    Fourier Transform
        ↓
    PSF
        ↓
    Optical Image
    """

    def __init__(
            self,
            coeffs,
            wavelength=550e-9,
            pupil_radius=1.0,
            psf_size=65,
            fft_size=512,
    ):

        self.phase = PhaseAberration(coeffs)

        self.wavelength = wavelength

        self.pupil_radius = pupil_radius

        self.psf_size = psf_size

        self.fft_size = fft_size

        self.kernel = self.get_kernel()

    # --------------------------------------------------------

    def build_coordinate(self):

        """
        Generate normalized optical coordinates.
        """

        size = self.fft_size

        y, x = np.mgrid[-1:1:complex(size),
                        -1:1:complex(size)]

        rho = np.sqrt(x ** 2 + y ** 2)

        theta = np.arctan2(y, x)

        return rho, theta

    # --------------------------------------------------------

    def pupil_function(self):

        """
        Circular pupil.
        """

        rho, theta = self.build_coordinate()

        aperture = (rho <= self.pupil_radius).astype(np.float64)

        phase = self.phase.generate(rho, theta)

        pupil = aperture * np.exp(
            1j * phase
        )

        return pupil

    # --------------------------------------------------------

    def compute_psf(self):

        """
        Fraunhofer diffraction.

        PSF = |FFT(P)|²
        """

        pupil = self.pupil_function()

        field = np.fft.fftshift(

            np.fft.fft2(

                np.fft.ifftshift(
                    pupil
                ),

                norm="ortho"

            )

        )

        psf = np.abs(field) ** 2

        psf /= psf.sum() + 1e-12

        return psf

    # --------------------------------------------------------

    @staticmethod
    def center_crop(image, crop_size):

        """
        Center crop.

        Never resize PSF.
        """

        h, w = image.shape

        cy = h // 2

        cx = w // 2

        half = crop_size // 2

        start_y = cy - crop_size // 2
        start_x = cx - crop_size // 2

        crop = image[
            start_y:start_y + crop_size,
            start_x:start_x + crop_size
        ]

        crop /= crop.sum() + 1e-12

        return crop

    # --------------------------------------------------------

    def get_kernel(self):

        """
        Final PSF kernel.
        """

        psf = self.compute_psf()

        kernel = self.center_crop(
            psf,
            self.psf_size
        )

        kernel = np.fft.ifftshift(kernel)

        return kernel.astype(np.float32)

    # --------------------------------------------------------

    @staticmethod
    def fft_convolution(image, kernel):

        """
        Frequency-domain convolution.
        """

        image = image.astype(np.float32)

        out = np.zeros_like(image)

        for c in range(image.shape[2]):

            out[:, :, c] = fftconvolve(

                image[:, :, c],

                kernel,

                mode="same"

            ).real

        out = np.clip(out, 0, 255)

        return out.astype(np.uint8)

    # --------------------------------------------------------

    @staticmethod
    def reflect_pad(image, pad):

        """
        Reflection padding.
        """

        return cv2.copyMakeBorder(

            image,

            pad,

            pad,

            pad,

            pad,

            borderType=cv2.BORDER_REFLECT101

        )

    # --------------------------------------------------------

    def forward(self, image):

        """
        Optical image formation.
        """
    # ---------- 输入检查 ----------
        assert image.ndim == 3, \
            "Input image must be H×W×C."

        assert image.shape[2] == 3, \
            "Input image must be RGB."
        
        # ---------- 生成 PSF ----------
        kernel = self.kernel

        pad = self.psf_size // 2

        padded = self.reflect_pad(image, pad)

        blurred = self.fft_convolution(
            padded,
            kernel
        )

        result = blurred[
                 pad:-pad,
                 pad:-pad
                 ]

        return result

    # --------------------------------------------------------

    def visualize_psf(self):

        psf = np.fft.fftshift(
            self.kernel.copy()
        )

        psf -= psf.min()

        psf /= psf.max() + 1e-12

        return (psf * 255).astype(np.uint8)
# ============================================================
#                 PP-OPPS PROCESSOR
# ============================================================

class PPOPPSProcessor:
    """
    High-level PP-OPPS processor.

    This class wraps the optical system and provides
    image-level processing APIs.
    """

    def __init__(
            self,
            preset="paper",
            coeffs=None,
            wavelength=550e-9,
            pupil_radius=1.0,
            psf_size=65,
            fft_size=512):

        # --------------------------------------------------
        # coefficients
        # --------------------------------------------------

        if coeffs is None:

            coeffs = paper_coefficients(preset)

        coeffs = np.asarray(coeffs, dtype=np.float64)

        if coeffs.ndim != 1:
            raise ValueError("coeffs must be a 1-D vector.")

        self.coeffs = coeffs

        # --------------------------------------------------
        # optical system
        # --------------------------------------------------

        self.optical = OpticalSystem(

            coeffs=self.coeffs,

            wavelength=wavelength,

            pupil_radius=pupil_radius,

            psf_size=psf_size,

            fft_size=fft_size

        )

    # --------------------------------------------------------

    @staticmethod
    def load_image(path):
        """
        Load RGB image.
        """

        image = cv2.imread(path, cv2.IMREAD_COLOR)

        if image is None:

            raise RuntimeError(
                f"Cannot read image:\n{path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        return image

    # --------------------------------------------------------

    @staticmethod
    def save_image(path, image):
        """
        Save RGB image.
        """

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        save_dir = os.path.dirname(path)

        if save_dir != "":
            os.makedirs(
                save_dir,
                exist_ok=True
            )

        cv2.imwrite(path, image)

    # --------------------------------------------------------

    def process(self, image):
        """
        Process one RGB image.

        Parameters
        ----------
        image : ndarray
            RGB uint8 image.

        Returns
        -------
        anonymized RGB image.
        """

        if image.ndim != 3:
            raise ValueError(
                f"Expect H×W×3 image, got {image.shape}"
            )

        if image.shape[2] != 3:
            raise ValueError(
                f"Expect RGB image, got {image.shape[2]} channels."
            )

        image = image.astype(np.uint8)

        result = self.optical.forward(image)

        return result

    # --------------------------------------------------------

    def process_file(
            self,
            input_path,
            output_path):
        """
        Process one image file.
        """

        image = self.load_image(input_path)

        result = self.process(image)

        self.save_image(
            output_path,
            result
        )

    # --------------------------------------------------------

    def get_psf(self):
        """
        Return normalized PSF image.

        Useful for visualization.
        """

        return self.optical.visualize_psf()

    # --------------------------------------------------------

    def save_psf(self, save_path):
        """
        Save PSF image.
        """

        psf = self.get_psf()

        save_dir = os.path.dirname(save_path)

        if save_dir != "":
            os.makedirs(
                save_dir,
                exist_ok=True
            )

        cv2.imwrite(save_path, psf)

    # --------------------------------------------------------

    def print_information(self):
        """
        Print processor configuration.
        """

        print("=" * 60)

        print("PP-OPPS Processor")

        print("=" * 60)

        print(f"PSF Size      : {self.optical.psf_size}")

        print(f"FFT Size      : {self.optical.fft_size}")

        print(f"Wavelength    : {self.optical.wavelength}")

        print(f"Pupil Radius  : {self.optical.pupil_radius}")

        print(f"Zernike Modes : {len(self.coeffs)}")

        print("=" * 60)

# ============================================================
#                  DATASET PROCESSING
# ============================================================

    def process_folder(
            self,
            input_dir,
            output_dir,
            suffix=(".jpg", ".jpeg", ".png", ".bmp")):
        """
        Batch process a dataset while preserving the directory
        structure.

        Parameters
        ----------
        input_dir : str
            Root directory of the original dataset.

        output_dir : str
            Root directory for saving anonymized images.

        suffix : tuple
            Supported image extensions.
        """

        if not os.path.exists(input_dir):
            raise FileNotFoundError(
                f"Input directory does not exist:\n{input_dir}"
            )

        image_list = []

        # --------------------------------------------------
        # collect image paths
        # --------------------------------------------------
        for root, _, files in os.walk(input_dir):

            for file in files:

                if file.lower().endswith(suffix):

                    image_list.append(
                        os.path.join(root, file)
                    )

        image_list.sort()

        print("=" * 60)
        print("PP-OPPS Batch Processing")
        print("=" * 60)
        print(f"Input Directory : {input_dir}")
        print(f"Output Directory: {output_dir}")
        print(f"Total Images    : {len(image_list)}")
        print("=" * 60)

        # --------------------------------------------------
        # process images
        # --------------------------------------------------
        for img_path in tqdm(
                image_list,
                ncols=100,
                desc="Processing"):

            # relative path
            relative_path = os.path.relpath(
                img_path,
                input_dir
            )

            save_path = os.path.join(
                output_dir,
                relative_path
            )

            try:

                self.process_file(
                    img_path,
                    save_path
                )

            except Exception as e:

                print(f"\nFailed: {img_path}")

                print(e)

        print("\nFinished!")
        print(f"Saved to:\n{output_dir}")

    # --------------------------------------------------------

    def process_single_directory(
            self,
            input_dir,
            output_dir):
        """
        Process only the current directory
        (no recursive traversal).

        This function is mainly for debugging.
        """

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        files = sorted(os.listdir(input_dir))

        for file in tqdm(
                files,
                desc="Processing"):

            if not file.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp")):
                continue

            src = os.path.join(
                input_dir,
                file
            )

            dst = os.path.join(
                output_dir,
                file
            )

            self.process_file(
                src,
                dst
            )
# ============================================================
#                         MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Input / Output
    # --------------------------------------------------------

    input_dir = r"/mnt/private/xwy/aed-mae/data/avenue/test/frames"

    output_dir = r"/mnt/private/xwy/SPAct/anonymization_training/ppopps_av2"

    # --------------------------------------------------------
    # Build PP-OPPS processor
    # --------------------------------------------------------

    processor = PPOPPSProcessor(

        preset="paper",

        wavelength=550e-9,

        pupil_radius=1.0,

        psf_size=65,

        fft_size=512

    )

    processor.print_information()

    # Optional:
    # save PSF image
    #
    # processor.save_psf("psf.png")

    # --------------------------------------------------------
    # Start
    # --------------------------------------------------------

    processor.process_folder(

        input_dir=input_dir,

        output_dir=output_dir

    )

    print("\nDone.")