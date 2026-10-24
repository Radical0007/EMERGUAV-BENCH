from __future__ import annotations

import math
import random
import shutil
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

_SEVERITY_STRENGTH = {
    "low": 0.25,
    "medium": 0.45,
    "high": 0.65,
    "critical": 0.85,
}


def _strength(severity: str) -> float:
    return _SEVERITY_STRENGTH.get(severity, 0.45)


def _save_image(image: Image.Image, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {}
    if target_path.suffix.lower() in {".jpg", ".jpeg"}:
        save_kwargs = {"quality": 88, "optimize": True}
    image.convert("RGB").save(target_path, **save_kwargs)


def _dust_veil(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    base = ImageEnhance.Contrast(image).enhance(max(0.35, 1.0 - strength * 0.55))
    haze = Image.new("RGB", image.size, (218, 199, 158))
    blended = Image.blend(base, haze, strength * 0.42)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    for _ in range(max(10, int(20 * strength))):
        y = rng.randint(0, height)
        alpha = rng.randint(10, int(35 + 40 * strength))
        draw.line([(0, y), (width, y + rng.randint(-20, 20))], fill=(230, 213, 176, alpha), width=rng.randint(3, 10))
    return Image.alpha_composite(blended.convert("RGBA"), overlay).convert("RGB").filter(ImageFilter.GaussianBlur(radius=strength))


def _heat_shimmer(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    del rng
    strength = _strength(severity)
    source = image.convert("RGB")
    width, height = source.size
    result = Image.new("RGB", source.size)
    amplitude = max(1, int(width * 0.006 * strength))
    period = max(12, int(height * 0.08))
    for y in range(height):
        offset = int(math.sin(y / period * math.tau) * amplitude)
        row = source.crop((0, y, width, y + 1))
        result.paste(row, (offset, y))
        if offset > 0:
            result.paste(row.crop((0, 0, offset, 1)), (0, y))
        elif offset < 0:
            result.paste(row.crop((width + offset, 0, width, 1)), (width + offset, y))
    return result.filter(ImageFilter.GaussianBlur(radius=0.4 + strength * 0.8))


def _high_glare(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    width, height = image.size
    overlay = Image.new("RGBA", image.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    center = (int(width * (0.25 + rng.random() * 0.5)), int(height * (0.15 + rng.random() * 0.4)))
    radius = int(min(width, height) * (0.18 + strength * 0.25))
    for step in range(radius, 0, -max(1, radius // 24)):
        alpha = int(220 * strength * (step / radius) ** 2)
        draw.ellipse(
            (center[0] - step, center[1] - step, center[0] + step, center[1] + step),
            fill=(255, 248, 210, alpha),
        )
    brightened = ImageEnhance.Brightness(image).enhance(1.0 + strength * 0.25)
    return Image.alpha_composite(brightened.convert("RGBA"), overlay).convert("RGB")


def _blur(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    del rng
    return image.filter(ImageFilter.GaussianBlur(radius=0.8 + _strength(severity) * 3.0))


def _small_target_long_distance(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    del rng
    strength = _strength(severity)
    width, height = image.size
    scale = max(0.45, 1.0 - strength * 0.38)
    small = image.resize((max(1, int(width * scale)), max(1, int(height * scale))), Image.Resampling.BICUBIC)
    canvas = Image.new("RGB", image.size, (118, 128, 128))
    canvas.paste(small, ((width - small.width) // 2, (height - small.height) // 2))
    return canvas.filter(ImageFilter.GaussianBlur(radius=strength * 0.7))


def _occlusion(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = result.size
    for _ in range(max(1, int(2 + strength * 5))):
        box_width = rng.randint(max(8, int(width * 0.08)), max(12, int(width * 0.24)))
        box_height = rng.randint(max(8, int(height * 0.08)), max(12, int(height * 0.24)))
        left = rng.randint(0, max(0, width - box_width))
        top = rng.randint(0, max(0, height - box_height))
        draw.rectangle((left, top, left + box_width, top + box_height), fill=(24, 28, 32))
    return result


def _noise(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    result = image.copy().convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    count = int(width * height * (0.00025 + strength * 0.0012))
    for _ in range(count):
        x = rng.randrange(width)
        y = rng.randrange(height)
        value = rng.choice([0, 255])
        alpha = int(55 + strength * 100)
        draw.point((x, y), fill=(value, value, value, alpha))
    return Image.alpha_composite(result.convert("RGBA"), overlay).convert("RGB")


def _partial_masking(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    result = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    mask_width = int(width * (0.18 + strength * 0.18))
    left = rng.randint(0, max(0, width - mask_width))
    draw.rectangle((left, 0, left + mask_width, height), fill=(10, 10, 10, int(95 + strength * 90)))
    return Image.alpha_composite(result, overlay).convert("RGB")


def _data_loss(image: Image.Image, severity: str, rng: random.Random) -> Image.Image:
    strength = _strength(severity)
    result = image.copy().convert("RGB")
    draw = ImageDraw.Draw(result)
    width, height = image.size
    stripe_count = max(2, int(3 + strength * 9))
    for _ in range(stripe_count):
        top = rng.randint(0, max(0, height - 2))
        stripe_height = rng.randint(2, max(3, int(height * (0.015 + strength * 0.04))))
        draw.rectangle((0, top, width, min(height, top + stripe_height)), fill=(0, 0, 0))
    block_count = max(1, int(1 + strength * 5))
    for _ in range(block_count):
        block_width = rng.randint(max(8, width // 16), max(12, width // 5))
        block_height = rng.randint(max(8, height // 16), max(12, height // 5))
        left = rng.randint(0, max(0, width - block_width))
        top = rng.randint(0, max(0, height - block_height))
        draw.rectangle((left, top, left + block_width, top + block_height), fill=(0, 0, 0))
    return result


_OPERATIONS = {
    "clean_visual": lambda image, severity, rng: image.copy().convert("RGB"),
    "dust_veil": _dust_veil,
    "heat_shimmer": _heat_shimmer,
    "high_glare": _high_glare,
    "blur": _blur,
    "small_target_long_distance": _small_target_long_distance,
    "occlusion": _occlusion,
    "noise": _noise,
    "partial_masking": _partial_masking,
    "data_loss": _data_loss,
}


def generate_degraded_image(
    source_path: str | Path,
    target_path: str | Path,
    *,
    degradation_label: str,
    severity: str,
    seed: int = 0,
) -> dict[str, Any]:
    source = Path(source_path)
    target = Path(target_path)
    if degradation_label == "clean_visual":
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        return {
            "degradation_label": degradation_label,
            "severity": severity,
            "seed": seed,
            "operation": "copy",
            "source_path": str(source),
            "target_path": str(target),
        }

    if degradation_label not in _OPERATIONS:
        raise KeyError(f"Unsupported degradation label: {degradation_label}")
    rng = random.Random(seed)
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    degraded = _OPERATIONS[degradation_label](image, severity, rng)
    _save_image(degraded, target)
    return {
        "degradation_label": degradation_label,
        "severity": severity,
        "seed": seed,
        "operation": degradation_label,
        "source_path": str(source),
        "target_path": str(target),
        "strength": _strength(severity),
    }
