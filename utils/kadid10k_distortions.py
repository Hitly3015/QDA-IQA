from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, Sequence, Union

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.interpolate import CubicSpline
from skimage import color, restoration


Array = np.ndarray
ImageInput = Union[Array, Image.Image, str, Path]


DISTORTION_NAMES: Dict[int, str] = {
    1: "gaussian_blur",
    2: "lens_blur",
    3: "motion_blur",
    4: "color_diffusion",
    5: "color_shift",
    6: "color_quantization",
    7: "color_saturation_1",
    8: "color_saturation_2",
    9: "jpeg2000_compression",
    10: "jpeg_compression",
    11: "gaussian_noise",
    12: "color_component_noise",
    13: "impulse_noise",
    14: "multiplicative_noise",
    15: "denoise",
    16: "brighten",
    17: "darken",
    18: "mean_shift",
    19: "jitter",
    20: "non_eccentricity_patch",
    21: "pixelate",
    22: "quantization_noise",
    23: "color_block",
    24: "high_sharpen",
    25: "contrast_change",
}


DISTORTION_LEVELS: Dict[int, Sequence[float]] = {
    1: (0.1, 0.5, 1, 2, 5),
    2: (1, 2, 4, 6, 8),
    3: (1, 2, 4, 6, 10),
    4: (1, 3, 6, 8, 12),
    5: (1, 3, 6, 8, 12),
    6: (64, 48, 32, 16, 8),
    7: (0.4, 0.2, 0.1, 0, -0.4),
    8: (1, 2, 3, 6, 9),
    9: (16, 32, 45, 120, 400),
    10: (43, 36, 24, 7, 4),
    11: (0.001, 0.002, 0.003, 0.005, 0.01),
    12: (0.0001, 0.0005, 0.001, 0.002, 0.003),
    13: (0.001, 0.005, 0.01, 0.02, 0.03),
    14: (0.001, 0.005, 0.01, 0.02, 0.05),
    15: (0.01, 0.03, 0.05, 0.1, 0.15),
    16: (0.1, 0.2, 0.4, 0.7, 1.1),
    17: (0.05, 0.1, 0.2, 0.4, 0.8),
    18: (0.15, 0.08, 0, -0.08, -0.15),
    19: (0.05, 0.1, 0.2, 0.5, 1),
    20: (20, 40, 60, 80, 100),
    21: (0.01, 0.05, 0.1, 0.2, 0.5),
    22: (20, 16, 13, 10, 7),
    23: (2, 4, 6, 8, 10),
    24: (1, 2, 3, 6, 12),
    25: (0.3, 0.15, 0, -0.4, -0.6),
}

_NAME_TO_ID = {name: idx for idx, name in DISTORTION_NAMES.items()}
_NAME_TO_ID.update({name.replace("_", " "): idx for idx, name in DISTORTION_NAMES.items()})


def _map_minmax(a: Array) -> Array:
    """MATLAB mapmm: map the global minimum and maximum to [0, 1]."""
    x = np.asarray(a, dtype=np.float64)
    lo, hi = float(np.min(x)), float(np.max(x))
    return (x - lo) / (hi - lo) if lo < hi else x.copy()


def _clip(a: Array) -> Array:
    return np.clip(a, 0.0, 1.0)


def _load_image(img: ImageInput) -> Array:
    if isinstance(img, (str, Path)):
        a = np.asarray(Image.open(img).convert("RGB"))
    elif isinstance(img, Image.Image):
        a = np.asarray(img.convert("RGB"))
    else:
        a = np.asarray(img)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=2)
        elif a.ndim == 3 and a.shape[2] == 1:
            a = np.repeat(a, 3, axis=2)
        elif a.ndim != 3 or a.shape[2] not in (3, 4):
            raise ValueError("img must have shape HxW, HxWx1, HxWx3, or HxWx4")
        if a.shape[2] == 4:
            a = a[..., :3]
    if not np.issubdtype(a.dtype, np.number):
        raise TypeError("img must contain numeric pixel values")
    return _map_minmax(a)


def _curves(image: Array, coefficient: Union[float, Sequence[float]]) -> Array:
    coef = np.asarray(coefficient, dtype=np.float64).ravel()
    if coef.size == 1:
        x = np.array([0.0, 0.5, 1.0])
        y = np.array([0.0, coef[0], 1.0])
    elif coef.size == 2:
        x = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
        y = np.array([0.0, coef[0], 0.5, coef[1], 1.0])
    else:
        raise ValueError("curve coefficient must contain one or two values")
    return _clip(CubicSpline(x, y)(np.asarray(image, dtype=np.float64)))


def _gaussian(image: Array, sigma: float) -> Array:
    sigmas = (sigma, sigma, 0) if image.ndim == 3 else sigma
    return ndimage.gaussian_filter(image, sigmas, mode="nearest")


def gaussian_blur(image: Array, radius: float) -> Array:
    return _gaussian(image, radius)


def _disk_kernel(radius: float) -> Array:
    radius = max(float(radius), 0.5)
    r = int(np.ceil(radius))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    # Supersampling gives a closer approximation to MATLAB fspecial('disk').
    samples = 7
    kernel = np.zeros_like(xx, dtype=np.float64)
    offsets = (np.arange(samples) + 0.5) / samples - 0.5
    for oy in offsets:
        for ox in offsets:
            kernel += ((xx + ox) ** 2 + (yy + oy) ** 2 <= radius ** 2)
    kernel /= kernel.sum()
    return kernel


def lens_blur(image: Array, radius: float) -> Array:
    kernel = _disk_kernel(radius)
    return ndimage.convolve(image, kernel[..., None], mode="nearest")


def _motion_kernel(length: float, angle: float) -> Array:
    length = max(1, int(round(length)))
    size = max(3, length if length % 2 else length + 1)
    kernel = np.zeros((size, size), dtype=np.float64)
    center = (size - 1) / 2
    theta = np.deg2rad(angle)
    half = (length - 1) / 2
    n = max(2, length * 8)
    xs = center + np.linspace(-half, half, n) * np.cos(theta)
    ys = center - np.linspace(-half, half, n) * np.sin(theta)
    for x, y in zip(xs, ys):
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        dx, dy = x - x0, y - y0
        for xi, yi, w in ((x0, y0, (1-dx)*(1-dy)),
                          (x0+1, y0, dx*(1-dy)),
                          (x0, y0+1, (1-dx)*dy),
                          (x0+1, y0+1, dx*dy)):
            if 0 <= xi < size and 0 <= yi < size:
                kernel[yi, xi] += w
    kernel /= kernel.sum()
    return kernel


def motion_blur(image: Array, radius: float, rng: np.random.Generator) -> Array:
    angle = int(rng.integers(0, 181))
    return ndimage.convolve(image, _motion_kernel(radius, angle)[..., None], mode="nearest")


def color_diffusion(image: Array, amount: float) -> Array:
    lab = color.rgb2lab(_clip(image))
    lab[..., 1:3] = _gaussian(lab[..., 1:3], 1.5 * amount + 2) * amount
    return _clip(color.lab2rgb(lab))


def _shift_replicate(channel: Array, shift: Sequence[int]) -> Array:
    return ndimage.shift(channel, shift=tuple(shift), order=0, mode="nearest", prefilter=False)


def color_shift(image: Array, amount: float, rng: np.random.Generator) -> Array:
    gray = color.rgb2gray(_clip(image))
    gy, gx = np.gradient(gray)
    edge = _map_minmax(_gaussian(np.sqrt(gx * gx + gy * gy), 4))
    lo, hi = np.percentile(edge, [0.1, 99.9])
    edge = _map_minmax(np.clip(edge, lo, hi))
    direction = rng.random(2)
    norm = np.linalg.norm(direction)
    shift = np.rint(direction / norm * amount).astype(int) if norm else np.zeros(2, int)
    shifted_green = _shift_replicate(image[..., 1], shift)
    result = image.copy()
    result[..., 1] = shifted_green * edge + image[..., 1] * (1 - edge)
    return result


def color_quantization(image: Array, colors: int) -> Array:
    pil = Image.fromarray(np.round(_clip(image) * 255).astype(np.uint8), "RGB")
    quantized = pil.quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT).convert("RGB")
    return np.asarray(quantized, dtype=np.float64) / 255.0


def color_saturation_1(image: Array, factor: float) -> Array:
    hsv = color.rgb2hsv(_clip(image))
    hsv[..., 1] *= factor
    return color.hsv2rgb(hsv)


def color_saturation_2(image: Array, factor: float) -> Array:
    lab = color.rgb2lab(_clip(image))
    lab[..., 1:3] *= factor
    return _clip(color.lab2rgb(lab))


def _encoded_roundtrip(image: Array, fmt: str, value: float) -> Array:
    pil = Image.fromarray(np.round(_clip(image) * 255).astype(np.uint8), "RGB")
    buffer = BytesIO()
    if fmt == "JPEG":
        pil.save(buffer, format="JPEG", quality=int(value), subsampling=2)
    else:
        # Pillow's JPEG2000 'rates' mode uses compression ratios, like MATLAB.
        pil.save(buffer, format="JPEG2000", quality_mode="rates", quality_layers=[float(value)])
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.float64) / 255.0


def jpeg2000_compression(image: Array, ratio: float) -> Array:
    try:
        return _encoded_roundtrip(image, "JPEG2000", ratio)
    except (OSError, ValueError) as exc:
        raise RuntimeError("Pillow was built without JPEG2000/OpenJPEG support") from exc


def jpeg_compression(image: Array, quality: float) -> Array:
    return _encoded_roundtrip(image, "JPEG", quality)


def gaussian_noise(image: Array, variance: float, rng: np.random.Generator) -> Array:
    return _clip(image + rng.normal(0, np.sqrt(variance), image.shape))


def color_component_noise(image: Array, variance: float, rng: np.random.Generator) -> Array:
    ycbcr = color.rgb2ycbcr(_clip(image))
    # skimage YCbCr uses video-range units; normalize each component before noise.
    scales = np.array([219.0, 224.0, 224.0])
    offsets = np.array([16.0, 128.0, 128.0])
    normalized = (ycbcr - offsets) / scales
    normalized = _clip(normalized + rng.normal(0, np.sqrt(variance), normalized.shape))
    return _clip(color.ycbcr2rgb(normalized * scales + offsets))


def impulse_noise(image: Array, density: float, rng: np.random.Generator) -> Array:
    result = image.copy()
    marks = rng.random(image.shape[:2])
    result[marks < density / 2] = 0
    result[(marks >= density / 2) & (marks < density)] = 1
    return result


def multiplicative_noise(image: Array, variance: float, rng: np.random.Generator) -> Array:
    return _clip(image + image * rng.normal(0, np.sqrt(variance), image.shape))


def denoise_distortion(image: Array, variance: float, rng: np.random.Generator) -> Array:
    """Add Gaussian noise and denoise it, mirroring the MATLAB DnCNN pipeline.

    The source uses MATLAB's pretrained DnCNN. To keep this file self-contained,
    this port uses non-local means with noise-aware parameters instead.
    """
    noisy = gaussian_noise(image, variance, rng)
    sigma = np.sqrt(variance)
    return restoration.denoise_nl_means(
        noisy, h=max(0.6 * sigma, 0.003), sigma=sigma,
        patch_size=5, patch_distance=6, fast_mode=True,
        channel_axis=-1, preserve_range=True,
    )


def brighten(image: Array, amount: float) -> Array:
    lab = color.rgb2lab(_clip(image))
    lab[..., 0] = _curves(lab[..., 0] / 100.0, 0.5 + amount / 2) * 100
    rgb_curve = _curves(image, 0.5 + amount / 2)
    return (2 * rgb_curve + _clip(color.lab2rgb(lab))) / 3


def darken(image: Array, amount: float) -> Array:
    return _curves(image, 0.5 - amount / 2)


def mean_shift(image: Array, amount: float) -> Array:
    return _clip(image + amount)


def _warp(image: Array, shifts: Array) -> Array:
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    coords = np.array([yy - shifts[..., 0], xx - shifts[..., 1]])
    result = np.empty_like(image)
    for channel in range(image.shape[2]):
        result[..., channel] = ndimage.map_coordinates(
            image[..., channel], coords, order=3, mode="nearest", prefilter=True
        )
    return result


def jitter(image: Array, amount: float, rng: np.random.Generator) -> Array:
    result = image.copy()
    for _ in range(5):
        shifts = rng.normal(0, amount, (*image.shape[:2], 2))
        result = _warp(result, shifts)
    return result


def non_eccentricity_patch(image: Array, count: int, rng: np.random.Generator) -> Array:
    result = image.copy()
    h, w = image.shape[:2]
    patch, radius = 16, 16
    if h < patch + 2 * radius or w < patch + 2 * radius:
        return result
    for _ in range(int(count)):
        y = int(rng.integers(radius, h - patch - radius + 1))
        x = int(rng.integers(radius, w - patch - radius + 1))
        block = result[y:y + patch, x:x + patch].copy()
        dy = int(np.rint((rng.random() - 0.5) * radius))
        dx = int(np.rint((rng.random() - 0.5) * radius))
        result[y + dy:y + dy + patch, x + dx:x + dx + patch] = block
    return result


def pixelate(image: Array, strength: float) -> Array:
    scale = max(0.02, 0.95 - strength ** 0.6)
    h, w = image.shape[:2]
    low_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    pil = Image.fromarray(np.round(_clip(image) * 255).astype(np.uint8), "RGB")
    small = pil.resize(low_size, Image.Resampling.NEAREST)
    return np.asarray(small.resize((w, h), Image.Resampling.NEAREST), dtype=np.float64) / 255


def quantization_noise(image: Array, levels: int) -> Array:
    # MATLAB uses global multi-Otsu thresholds; a uniform scalar quantizer is a
    # stable, efficient equivalent at the large threshold counts used here.
    levels = int(levels)
    return np.round(_clip(image) * levels) / levels


def color_block(image: Array, count: int, rng: np.random.Generator) -> Array:
    result = image.copy()
    h, w = image.shape[:2]
    ph, pw = min(32, h), min(32, w)
    for _ in range(int(count)):
        y = int(rng.integers(0, h - ph + 1))
        x = int(rng.integers(0, w - pw + 1))
        result[y:y + ph, x:x + pw] = rng.random(3)
    return result


def high_sharpen(image: Array, amount: float, radius: float = 3) -> Array:
    # MATLAB imsharpen defaults to Threshold=0 and uses an unsharp mask.
    return _clip(image + amount * (image - _gaussian(image, radius)))


def contrast_change(image: Array, amount: float) -> Array:
    return _curves(image, (0.25 - amount / 4, 0.75 + amount / 4))


def _resolve_distortion(distortion_type: Union[int, str]) -> int:
    if isinstance(distortion_type, str):
        key = distortion_type.strip().lower().replace("-", "_")
        if key.isdigit():
            distortion_type = int(key)
        else:
            key = key.replace(" ", "_")
            if key not in _NAME_TO_ID:
                valid = ", ".join(f"{i}:{n}" for i, n in DISTORTION_NAMES.items())
                raise ValueError(f"unknown distortion type {distortion_type!r}; valid types: {valid}")
            distortion_type = _NAME_TO_ID[key]
    if isinstance(distortion_type, bool) or not isinstance(distortion_type, (int, np.integer)):
        raise TypeError("distortion_type must be an integer ID or a distortion name")
    distortion_type = int(distortion_type)
    if distortion_type not in DISTORTION_NAMES:
        raise ValueError("distortion_type must be in the range 1..25")
    return distortion_type


def apply_distortion(
    img: ImageInput,
    distortion_type: Union[int, str],
    distortion_level: int,
    seed: Union[int, None] = None,
) -> Array:
    """Apply one KADID-10k distortion.

    Parameters
    ----------
    img:
        NumPy image, PIL image, or image path.
    distortion_type:
        Integer 1..25 or a name from ``DISTORTION_NAMES``.
    distortion_level:
        Severity level 1..5 (1 is weakest, 5 is strongest).
    seed:
        Optional seed for reproducible random distortions.

    Returns
    -------
    numpy.ndarray
        RGB float32 image in [0, 1].
    """
    dist_id = _resolve_distortion(distortion_type)
    if isinstance(distortion_level, bool) or not isinstance(distortion_level, (int, np.integer)):
        raise TypeError("distortion_level must be an integer from 1 to 5")
    if not 1 <= int(distortion_level) <= 5:
        raise ValueError("distortion_level must be in the range 1..5")

    image = _load_image(img)
    value = DISTORTION_LEVELS[dist_id][int(distortion_level) - 1]
    rng = np.random.default_rng(seed)

    functions: Dict[int, Callable[[], Array]] = {
        1: lambda: gaussian_blur(image, value),
        2: lambda: lens_blur(image, value),
        3: lambda: motion_blur(image, value, rng),
        4: lambda: color_diffusion(image, value),
        5: lambda: color_shift(image, value, rng),
        6: lambda: color_quantization(image, int(value)),
        7: lambda: color_saturation_1(image, value),
        8: lambda: color_saturation_2(image, value),
        9: lambda: jpeg2000_compression(image, value),
        10: lambda: jpeg_compression(image, value),
        11: lambda: gaussian_noise(image, value, rng),
        12: lambda: color_component_noise(image, value, rng),
        13: lambda: impulse_noise(image, value, rng),
        14: lambda: multiplicative_noise(image, value, rng),
        15: lambda: denoise_distortion(image, value, rng),
        16: lambda: brighten(image, value),
        17: lambda: darken(image, value),
        18: lambda: mean_shift(image, value),
        19: lambda: jitter(image, value, rng),
        20: lambda: non_eccentricity_patch(image, int(value), rng),
        21: lambda: pixelate(image, value),
        22: lambda: quantization_noise(image, int(value)),
        23: lambda: color_block(image, int(value), rng),
        24: lambda: high_sharpen(image, value),
        25: lambda: contrast_change(image, value),
    }
    # The original generator applies mapmm both before and after distortion.
    return _map_minmax(functions[dist_id]()).astype(np.float32)


def to_uint8(image: Array) -> Array:
    """Convert the generator's floating-point output to saveable uint8 RGB."""
    return np.round(_clip(image) * 255).astype(np.uint8)


def imdist_generator(
    img: ImageInput,
    dist_type: Union[int, str],
    dist_level: int,
    seed: Union[int, None] = None,
) -> Array:
    """MATLAB-compatible unified call: image + type + severity level."""
    return apply_distortion(img, dist_type, dist_level, seed=seed)


__all__ = [
    "DISTORTION_NAMES", "DISTORTION_LEVELS", "apply_distortion",
    "imdist_generator", "to_uint8",
]
