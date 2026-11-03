"""Acquisition corruptions (benchmark) and train-time augmentation.

Each corruption: f(img uint8 RGB HxWx3, severity 1..5, rng=None) -> uint8, calibrated on 256 px images.
Seen (also used by `robust` augmentation): brightness, gaussian_blur, gaussian_noise, hair.
Unseen (test only): motion_blur, jpeg.

Usage:
    python -m src.corruptions        # writes results/figures/corruptions_grid.png
"""
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

SEEN = ["brightness", "gaussian_blur", "gaussian_noise", "hair"]
UNSEEN = ["motion_blur", "jpeg"]


def _rng(rng):
    return rng if rng is not None else np.random.default_rng(0)


def brightness(img, severity, rng=None):
    """Over- or under-exposure (the sign is drawn from rng)."""
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32) / 255
    hsv[..., 2] = hsv[..., 2] + c if _rng(rng).random() < 0.5 else hsv[..., 2] * (1 - c)
    return cv2.cvtColor((np.clip(hsv, 0, 1) * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)


def gaussian_blur(img, severity, rng=None):
    sigma = [1, 2, 3, 4, 6][severity - 1]
    return cv2.GaussianBlur(img, (0, 0), sigma)


def gaussian_noise(img, severity, rng=None):
    std = [0.03, 0.06, 0.09, 0.13, 0.18][severity - 1] * 255
    return np.clip(img + _rng(rng).normal(0, std, img.shape), 0, 255).astype(np.uint8)


def hair(img, severity, rng=None):
    """Thin dark curves (quadratic Bezier) drawn over the image."""
    rng = _rng(rng)
    n = [3, 6, 10, 16, 24][severity - 1]
    h, w = img.shape[:2]
    out = img.copy()
    t = np.linspace(0, 1, 30)[:, None]
    for _ in range(n):
        p0, p1, p2 = rng.uniform(0, 1, (3, 2)) * [w, h]
        pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2
        color = tuple(int(v) for v in rng.integers(10, 60) * np.array([1.0, 0.8, 0.7]))
        cv2.polylines(out, [pts.astype(np.int32)], False, color, int(rng.integers(1, 3)), cv2.LINE_AA)
    return out


def motion_blur(img, severity, rng=None):
    k = [5, 9, 13, 17, 23][severity - 1]
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2] = 1
    rot = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), _rng(rng).uniform(0, 180), 1)
    kernel = cv2.warpAffine(kernel, rot, (k, k))
    return cv2.filter2D(img, -1, kernel / kernel.sum())


def jpeg(img, severity, rng=None):
    q = [50, 30, 20, 15, 10][severity - 1]
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


CORRUPTIONS = {f.__name__: f for f in (brightness, gaussian_blur, gaussian_noise, hair, motion_blur, jpeg)}


def apply_corruption(img, name, severity, img_id):
    """Deterministic: RNG seeded with (img_id, corruption, severity)."""
    rng = np.random.default_rng(zlib.crc32(f"{img_id}|{name}|{severity}".encode()))
    return CORRUPTIONS[name](img, severity, rng)


class RobustAug:
    """Random photometric changes + seen corruptions, applied on the full image before cropping."""

    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, pil):
        # Seed from torch so DataLoader seeding makes it reproducible.
        rng = np.random.default_rng(int(torch.randint(2**31, (1,))))
        img = np.asarray(pil, np.float32) / 255
        if rng.random() < 0.5:  # gamma
            img = img ** rng.uniform(0.7, 1.5)
        if rng.random() < 0.5:  # color temperature: warm/cool shift of R vs B
            t = rng.uniform(-0.1, 0.1)
            img = img * [1 + t, 1, 1 - t]
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
        for name in SEEN:
            if rng.random() < self.p:
                img = CORRUPTIONS[name](img, int(rng.integers(1, 6)), rng)
        return Image.fromarray(img)


def build_transform(aug, train, img_size=224):
    """Tensor in [0, 1]; ImageNet normalization lives in the model (src.train.Net)."""
    if not train or aug == "none":
        return T.Compose([T.Resize(img_size), T.CenterCrop(img_size), T.ToTensor()])
    ops = [RobustAug(), T.ColorJitter(0.3, 0.3, 0.15, 0.02)] if aug == "robust" else []
    assert aug in ("standard", "robust"), aug
    return T.Compose(ops + [
        T.RandomRotation(90),
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ToTensor(),
    ])


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd

    split = pd.read_csv("data/splits/pad_split.csv")
    img_id = split.query("split == 'test' and diagnostic == 'MEL'")["img_id"].iloc[0]
    img = np.asarray(Image.open(next(Path("data/raw/pad_256").rglob(img_id))).convert("RGB"))

    for name in CORRUPTIONS:
        a, b = apply_corruption(img, name, 3, img_id), apply_corruption(img, name, 3, img_id)
        assert a.dtype == np.uint8 and a.shape == img.shape and np.array_equal(a, b), name
    x = build_transform("robust", True)(Image.fromarray(img))
    assert x.shape == (3, 224, 224) and 0 <= x.min() and x.max() <= 1

    fig, axes = plt.subplots(len(CORRUPTIONS), 6, figsize=(12, 2.1 * len(CORRUPTIONS)))
    for r, name in enumerate(CORRUPTIONS):
        for s in range(6):
            ax = axes[r, s]
            ax.imshow(img if s == 0 else apply_corruption(img, name, s, img_id))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title("limpia" if s == 0 else f"severidad {s}")
        axes[r, 0].set_ylabel(name + ("" if name in SEEN else "\n(no vista)"))
    plt.tight_layout()
    plt.savefig("results/figures/corruptions_grid.png", dpi=120)
    print("ok: deterministic, results/figures/corruptions_grid.png")
