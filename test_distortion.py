#!/usr/bin/env python3
"""Проверка детектора искажений на всех скриншотах из папки."""
import cv2
import numpy as np
from pathlib import Path

LAP_THRESH = 800.0
PERIOD_THRESH = 60.0
MAX_SAT = 0.05
DARK_BRIGHT = 0.15
DARK_LAP = 1200.0


def is_distorted(img_bgr: np.ndarray) -> tuple[bool, dict]:
    """Два типа: 1) тёмный+низкая резкость, 2) ЧБ+периодичность+низкая резкость."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = np.mean(gray) / 255.0
    sat_mean = np.mean(hsv[:, :, 1]) / 255.0

    if brightness < DARK_BRIGHT and laplacian_var < DARK_LAP:
        return True, {"lap": laplacian_var, "bright": brightness, "reason": "dark"}

    row_means = np.mean(gray, axis=1).astype(np.float64)
    fft = np.fft.rfft(row_means - np.mean(row_means))
    magnitude = np.abs(fft)
    mag_no_dc = magnitude[1:] if len(magnitude) > 1 else np.array([0.0])
    mean_mag = np.mean(mag_no_dc) if len(mag_no_dc) > 0 else 0
    max_mag = np.max(mag_no_dc) if len(mag_no_dc) > 0 else 0
    ratio = (max_mag / mean_mag) if mean_mag > 1e-6 else 0
    low_sharpness = laplacian_var < LAP_THRESH
    strong_periodicity = ratio > PERIOD_THRESH
    is_grayscale = sat_mean < MAX_SAT
    bad = low_sharpness and strong_periodicity and is_grayscale
    return bad, {"lap": laplacian_var, "period_ratio": ratio, "sat": sat_mean, "reason": "grayscale_periodic"}


def main():
    folder = Path("camera_screenshots")
    if not folder.exists():
        print("Папка camera_screenshots не найдена")
        return
    files = sorted(folder.glob("*.jpg"))
    distorted_list = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            continue
        bad, m = is_distorted(img)
        if bad:
            distorted_list.append((p.name, m))

    print(f"Искажённые (тёмный bright<{DARK_BRIGHT} или ЧБ+периодичность):")
    for name, m in distorted_list:
        r = m.get("reason", "")
        if r == "dark":
            print(f"  {name}: dark (bright={m['bright']:.3f}, lap={m['lap']:.0f})")
        else:
            print(f"  {name}: grayscale (lap={m['lap']:.0f}, period={m['period_ratio']:.1f}, sat={m['sat']:.3f})")
    print(f"Всего: {len(distorted_list)} из {len(files)}")


if __name__ == "__main__":
    main()
