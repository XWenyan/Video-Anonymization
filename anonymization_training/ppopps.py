import numpy as np
import cv2
from scipy.special import factorial
from scipy.ndimage import convolve
import os
from tqdm import tqdm

# ==================== Zernike 与 PSF 相关函数 ====================
def radial_zernike(n, m, rho):
    m_abs = abs(m)
    if (n - m_abs) % 2 != 0:
        return np.zeros_like(rho)
    radial = np.zeros_like(rho)
    for k in range((n - m_abs) // 2 + 1):
        coeff = (-1) ** k * factorial(n - k)
        coeff /= factorial(k) * factorial((n + m_abs) // 2 - k) * factorial((n - m_abs) // 2 - k)
        radial += coeff * (rho ** (n - 2 * k))
    return radial

def zernike_polynomial(n, m, rho, theta):
    radial = radial_zernike(n, m, rho)
    if m > 0:
        return radial * np.cos(m * theta)
    elif m < 0:
        return radial * np.sin(abs(m) * theta)
    else:
        return radial

def wavefront_aberration(z_coeffs, rho, theta):
    """计算波前像差 W(rho, theta) = sum(z_coeffs[i] * Z_i(rho, theta))"""
    noll_indices = [(1, 1), (2, -1), (2, 1), (3, -2), (3, 0), (3, 2), (4, -3),
                    (4, -1), (4, 1), (4, 3), (5, -4), (5, -2), (5, 0), (5, 2), (5, 4)]
    w = np.zeros_like(rho)
    for idx, coeff in enumerate(z_coeffs):
        if coeff == 0 or idx >= len(noll_indices):
            continue
        n, m = noll_indices[idx]
        w += coeff * zernike_polynomial(n, m, rho, theta)
    return w

def compute_psf_from_wavefront(w, rho_normalized, wavelength=550e-9, pixel_size=3.69e-6, f=35.5e-3):
    """从波前像差计算PSF"""
    w_radial = w * wavelength
    aperture = np.ones_like(w)
    aperture[rho_normalized > 1] = 0   # 圆形孔径
    amplitude = aperture * np.exp(1j * 2 * np.pi * w_radial / wavelength)
    psf = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.fftshift(amplitude)))) ** 2
    psf = psf / psf.sum()
    return psf

def generate_pp_opps_image(image, z_coeffs, wavelength=550e-9, pixel_size=3.69e-6, f=35.5e-3, kernel_size=31):
    """对单张图像应用 PP-OPPS 模糊"""
    h, w = image.shape[:2]
    y, x = np.ogrid[-h//2:h//2, -w//2:w//2]
    r = np.sqrt(x**2 + y**2)
    max_r = min(h, w) // 2
    rho = r / max_r
    theta = np.arctan2(y, x)
    rho_normalized = np.clip(rho, 0, 1)

    # 波前像差
    w = wavefront_aberration(z_coeffs, rho_normalized, theta)

    # PSF
    psf = compute_psf_from_wavefront(w, rho_normalized, wavelength, pixel_size, f)

    # 调整尺寸
    if kernel_size != psf.shape[0]:
        psf = cv2.resize(psf, (kernel_size, kernel_size), interpolation=cv2.INTER_LINEAR)

    # 卷积
    result = np.zeros_like(image, dtype=np.float32)
    for c in range(3):
        result[:, :, c] = convolve(image[:, :, c].astype(np.float32), psf, mode='constant')
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def generate_psf(z_coeffs, size=31, wavelength=550e-9, pixel_size=3.69e-6, f=35.5e-3):
    """单独生成 PSF 核，用于调试或可视化"""
    y, x = np.ogrid[-size//2:size//2, -size//2:size//2]
    r = np.sqrt(x**2 + y**2)
    max_r = size // 2
    rho = r / max_r
    theta = np.arctan2(y, x)
    rho_normalized = np.clip(rho, 0, 1)
    w = wavefront_aberration(z_coeffs, rho_normalized, theta)
    psf = compute_psf_from_wavefront(w, rho_normalized, wavelength, pixel_size, f)
    return psf

# ==================== 预设像差系数 ====================
def pp_opps_preset(preset='defocus'):
    """预设的 Zernike 系数（前15个，按 Noll 索引）"""
    coeffs = [0] * 15
    if preset == 'defocus':
        coeffs[4] = 2.5      # Z4: defocus
    elif preset == 'astigmatism':
        coeffs[3] = 1.5
        coeffs[4] = 2.0
        coeffs[5] = 1.5
    elif preset == 'coma':
        coeffs[4] = 2.0
        coeffs[7] = 1.2
        coeffs[8] = 1.2
    elif preset == 'full':
        coeffs[3] = 1.0
        coeffs[4] = 1.5
        coeffs[5] = 1.0
        coeffs[7] = 0.5
        coeffs[8] = 0.5
        coeffs[12] = 0.3
    elif preset == 'paper_initial':
        coeffs[4] = -51.0   # 论文初始化值
    return coeffs

# ==================== 批量处理与主函数 ====================
def batch_process(input_dir, output_dir, preset='defocus', kernel_size=31):
    """批量处理整个文件夹的图像"""
    os.makedirs(output_dir, exist_ok=True)

    image_ext = ('.png', '.jpg', '.jpeg', '.bmp')
    image_files = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if f.lower().endswith(image_ext):
                image_files.append(os.path.join(root, f))

    if not image_files:
        print(f"未找到图像文件: {input_dir}")
        return

    z_coeffs = pp_opps_preset(preset)
    print(f"使用预设: {preset}, Zernike系数: {z_coeffs}")

    for img_path in tqdm(image_files, desc="PP-OPPS 处理"):
        img = cv2.imread(img_path)
        if img is None:
            continue

        processed = generate_pp_opps_image(img, z_coeffs, kernel_size=kernel_size)

        # 保持相对路径结构
        rel_path = os.path.relpath(img_path, input_dir)
        out_path = os.path.join(output_dir, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, processed)

    print(f"处理完成，结果保存在: {output_dir}")

def compute_psnr_batch(orig_dir, proc_dir):
    """计算原始与处理后图像的 PSNR 统计"""
    def compute_psnr(img1, img2, max_pixel=255.0):
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 10 * np.log10((max_pixel ** 2) / mse)

    pairs = []
    for root, _, files in os.walk(orig_dir):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                orig_path = os.path.join(root, f)
                rel = os.path.relpath(orig_path, orig_dir)
                proc_path = os.path.join(proc_dir, rel)
                if os.path.exists(proc_path):
                    pairs.append((orig_path, proc_path))

    if not pairs:
        print("未找到匹配的图像对")
        return

    psnr_vals = []
    for orig_path, proc_path in tqdm(pairs, desc="计算 PSNR"):
        img_orig = cv2.imread(orig_path)
        img_proc = cv2.imread(proc_path)
        if img_orig is None or img_proc is None:
            continue
        if img_orig.shape != img_proc.shape:
            img_proc = cv2.resize(img_proc, (img_orig.shape[1], img_orig.shape[0]))
        psnr = compute_psnr(img_orig, img_proc)
        psnr_vals.append(psnr)

    psnr_arr = np.array(psnr_vals)
    print("\n====== PSNR 统计 (原始 vs PP-OPPS) ======")
    print(f"平均 PSNR: {np.mean(psnr_arr):.2f} dB")
    print(f"中位数 PSNR: {np.median(psnr_arr):.2f} dB")
    print(f"标准差: {np.std(psnr_arr):.2f} dB")
    print(f"最小值: {np.min(psnr_arr):.2f} dB")
    print(f"最大值: {np.max(psnr_arr):.2f} dB")

# ==================== 主函数 ====================
def main():
    # 请根据实际情况修改以下路径
    input_dir = "/mnt/private/xwy/aed-mae/data/shanghaitech_AED/test/frames"    # 原始图像文件夹
    output_dir = "/mnt/private/xwy/SPAct/anonymization_training/ppopps_sh"    # PP-OPPS 输出文件夹

    # 可选预设: 'defocus', 'astigmatism', 'coma', 'full', 'paper_initial'
    preset = 'defocus'
    kernel_size = 31   # PSF 核大小，可根据需要调整

    # 1. 生成 PP-OPPS 图像
    batch_process(input_dir, output_dir, preset=preset, kernel_size=kernel_size)

    # 2. 计算 PSNR 统计
    compute_psnr_batch(input_dir, output_dir)

if __name__ == "__main__":
    main()