"""Fast PSNR/SSIM that reproduce skimage defaults (win 7, uniform filter,
sample covariance, K1=.01, K2=.03, data_range=255, border crop 3)."""
import numpy as np, cv2

def fast_psnr(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    mse = np.mean((a - b) ** 2)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(255.0 ** 2 / mse))

def fast_ssim(a, b, win=7, data_range=255.0):
    X = a.astype(np.float64); Y = b.astype(np.float64)
    k = (win, win)
    blur = lambda z: cv2.blur(z, k, borderType=cv2.BORDER_REFLECT)
    ux, uy = blur(X), blur(Y)
    uxx, uyy, uxy = blur(X * X), blur(Y * Y), blur(X * Y)
    NP = win * win; cov_norm = NP / (NP - 1.0)
    vx = cov_norm * (uxx - ux * ux); vy = cov_norm * (uyy - uy * uy); vxy = cov_norm * (uxy - ux * uy)
    C1 = (0.01 * data_range) ** 2; C2 = (0.03 * data_range) ** 2
    S = ((2 * ux * uy + C1) * (2 * vxy + C2)) / ((ux * ux + uy * uy + C1) * (vx + vy + C2))
    p = (win - 1) // 2
    return float(S[p:-p, p:-p].mean())

def reconstruct(gray, thresholds, mode="even"):
    """mode 'even': current apply_thresholds (0,64,128,191,255). 'mean': class means."""
    th = sorted(int(np.clip(round(t), 0, 255)) for t in thresholds)
    bounds = [0] + th + [256]
    n = len(bounds) - 1
    lut = np.zeros(256, dtype=np.uint8)
    if mode == "even":
        for i, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:])):
            lut[lo:hi] = int(round(255 * i / max(1, n - 1))) if n > 1 else 255
    elif mode == "mean":
        hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
        lv = np.arange(256, dtype=np.float64)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi <= lo: continue
            m = hist[lo:hi].sum()
            mu = (hist[lo:hi] * lv[lo:hi]).sum() / m if m > 0 else (lo + hi - 1) / 2.0
            lut[lo:hi] = int(round(mu))
    else:
        raise ValueError(mode)
    return lut[gray]
