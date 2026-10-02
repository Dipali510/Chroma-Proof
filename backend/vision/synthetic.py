"""Synthetic DEMO image generator (no real/sensitive data).

Draws the demo reference card (see config.py) with a reagent window, places it
on a textured background with perspective, simulates a colour cast and sensor
noise, and offers helpers to degrade the image (blur, glare, darkness...).
Used by the tests. Run  ``python -m vision.synthetic [out_dir]``  to write the demo images
(sample_data/{good,blurry,glare,missing_card,borderline}) and example_outputs.json.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import cv2
import numpy as np

from .config import DEFAULT_CONFIG, ChromaProofConfig

POSITIVE_RGB = DEFAULT_CONFIG.classification.positive_reference_rgb
NEGATIVE_RGB = DEFAULT_CONFIG.classification.negative_reference_rgb
BORDERLINE_RGB = tuple(int(round(0.75 * p + 0.25 * n)) for p, n in zip(POSITIVE_RGB, NEGATIVE_RGB))
UNSUPPORTED_RGB = (60, 170, 80)     # green: not one of the configured outcomes

CANVAS = (800, 600)                  # (width, height)
DEFAULT_QUAD = [[110, 100], [660, 115], [648, 470], [100, 460]]   # card corners TL,TR,BR,BL


def render_card(reagent_rgb, cfg: ChromaProofConfig = DEFAULT_CONFIG,
                occluded_patches: Sequence[int] = ()) -> np.ndarray:
    """Draw the flat, front-on demo card (BGR uint8) at ``cfg.card.warp_size``."""
    W, H = cfg.card.warp_size
    card = np.zeros((H, W, 3), np.uint8)                      # near-black border
    card[...] = (15, 15, 15)
    b = int(0.04 * H)
    card[b:H - b, b:W - b] = (208, 215, 215)                  # BGR of (215,215,208) card body
    for i, spec in enumerate(cfg.card.patches):
        cx, cy = spec.center[0] * W, spec.center[1] * H
        hw, hh = spec.half_size[0] * W, spec.half_size[1] * H
        p0, p1 = (int(cx - hw), int(cy - hh)), (int(cx + hw), int(cy + hh))
        colour = (224, 172, 105)[::-1] if i in occluded_patches else spec.expected_rgb[::-1]
        cv2.rectangle(card, p0, p1, colour, -1)
    x0, y0, x1, y1 = cfg.roi.window
    p0, p1 = (int(x0 * W), int(y0 * H)), (int(x1 * W), int(y1 * H))
    cv2.rectangle(card, p0, p1, (90, 90, 90), 3)              # window frame
    cv2.rectangle(card, (p0[0] + 3, p0[1] + 3), (p1[0] - 3, p1[1] - 3), tuple(reagent_rgb[::-1]), -1)
    return card


def render_scene(reagent_rgb=POSITIVE_RGB, *, with_card: bool = True,
                 quad: Optional[Sequence[Sequence[float]]] = None,
                 lighting_gain=(0.92, 0.88, 0.82), lighting_offset=(4, 2, 0),
                 noise_sigma: float = 5.0, seed: int = 0,
                 occluded_patches: Sequence[int] = (),
                 cfg: ChromaProofConfig = DEFAULT_CONFIG) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return (BGR image, homography card->image). ``lighting_*`` are given as (R, G, B)."""
    rng = np.random.default_rng(seed)
    cw, ch = CANVAS
    yy, xx = np.mgrid[0:ch, 0:cw].astype(np.float32)
    base = np.array([160, 172, 168], np.float32)               # BGR of a light grey-green desk
    img = base[None, None, :] + (12 * xx / cw + 8 * yy / ch)[..., None]
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (ch, cw)).astype(np.float32), (0, 0), 40) * 120
    img = img + blotch[..., None]

    H = None
    if with_card:
        card = render_card(reagent_rgb, cfg, occluded_patches)
        W0, H0 = cfg.card.warp_size
        src = np.array([[0, 0], [W0 - 1, 0], [W0 - 1, H0 - 1], [0, H0 - 1]], np.float32)
        dst = np.array(quad if quad is not None else DEFAULT_QUAD, np.float32)
        H = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(card, H, (cw, ch), flags=cv2.INTER_LINEAR).astype(np.float32)
        mask = cv2.warpPerspective(np.ones((H0, W0), np.float32), H, (cw, ch), flags=cv2.INTER_LINEAR)
        img = img * (1 - mask[..., None]) + warped * mask[..., None]
    else:  # no card: just a coloured swatch lying on the desk
        cv2.rectangle(img, (300, 220), (480, 380), tuple(float(c) for c in reagent_rgb[::-1]), -1)

    gain_bgr = np.array(lighting_gain[::-1], np.float32)
    off_bgr = np.array(lighting_offset[::-1], np.float32)
    img = img * gain_bgr + off_bgr
    img = img + rng.normal(0, noise_sigma, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8), H


# --- degradations ---------------------------------------------------------- #
def blur(img: np.ndarray, sigma: float = 4.0) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def darken(img: np.ndarray, factor: float = 0.25) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def add_glare(img: np.ndarray, center_xy: Tuple[float, float], radius: int = 60) -> np.ndarray:
    """Paint a soft-edged saturated white blob (specular highlight)."""
    mask = np.zeros(img.shape[:2], np.float32)
    cv2.circle(mask, (int(center_xy[0]), int(center_xy[1])), radius, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (0, 0), 3)[..., None]
    return np.clip(img * (1 - mask) + 255.0 * mask, 0, 255).astype(np.uint8)


def card_point_to_image(H: np.ndarray, fx: float, fy: float, cfg: ChromaProofConfig = DEFAULT_CONFIG):
    W0, H0 = cfg.card.warp_size
    p = H @ np.array([fx * W0, fy * H0, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def roi_center_in_image(H: np.ndarray, cfg: ChromaProofConfig = DEFAULT_CONFIG):
    x0, y0, x1, y1 = cfg.roi.window
    return card_point_to_image(H, 0.5 * (x0 + x1), 0.5 * (y0 + y1), cfg)


# --- demo-data generator ---------------------------------------------------- #
def generate_samples(out_dir: str = "sample_data") -> dict:
    """Write synthetic demo images into ``out_dir`` sub-folders plus example_outputs.json."""
    import json
    import os

    from .processor import process_image

    pos, H = render_scene(POSITIVE_RGB, seed=1)
    items = [
        ("good", "positive_demo.jpg", pos),
        ("good", "negative_demo.jpg", render_scene(NEGATIVE_RGB, seed=2, lighting_gain=(1.0, 0.95, 0.85))[0]),
        ("blurry", "blurry_demo.jpg", blur(pos, 4.0)),
        ("glare", "glare_demo.jpg", add_glare(pos, roi_center_in_image(H), 60)),
        ("missing_card", "missing_card_demo.jpg", render_scene(POSITIVE_RGB, with_card=False, seed=3)[0]),
        ("borderline", "borderline_demo.jpg", render_scene(BORDERLINE_RGB, seed=4)[0]),
        ("borderline", "unsupported_colour_demo.jpg", render_scene(UNSUPPORTED_RGB, seed=5)[0]),
    ]
    outputs = {}
    for folder, name, img in items:
        os.makedirs(os.path.join(out_dir, folder), exist_ok=True)
        path = os.path.join(out_dir, folder, name)
        cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        outputs[f"{folder}/{name}"] = process_image(path)
    with open(os.path.join(out_dir, "example_outputs.json"), "w") as fh:
        json.dump(outputs, fh, indent=2)
    return outputs


if __name__ == "__main__":
    import sys
    for key, val in generate_samples(sys.argv[1] if len(sys.argv) > 1 else "sample_data").items():
        print(f"{key:42s} {val['quality_status']:7s} {val['result']:13s} {val['confidence']}")
