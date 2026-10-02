"""Image-quality gate and reference-card detection.

Global checks (blur, exposure, glare) run on a down-scaled copy of the image.
Card detection looks for a card-shaped quadrilateral (dark border / strong edges),
validates its shape and returns its four corners.

All thresholds come from ``config.py`` (prototype/demo values).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from .config import ChromaProofConfig
from .models import CardDetection, QualityMetrics, glare_mask_rgb


# --------------------------------------------------------------------------- #
# Global quality metrics
# --------------------------------------------------------------------------- #
def downscale(img_bgr: np.ndarray, max_side: int) -> Tuple[np.ndarray, float]:
    """Shrink so the longest side <= max_side. Returns (image, scale<=1)."""
    h, w = img_bgr.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        img_bgr = cv2.resize(img_bgr, (max(1, round(w * scale)), max(1, round(h * scale))),
                             interpolation=cv2.INTER_AREA)
    return img_bgr, scale


def compute_blur_score(gray: np.ndarray) -> float:
    """Variance of the Laplacian: low value = few sharp edges = blurry."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_glare_fraction(img_bgr: np.ndarray, v_min: int, s_max: int) -> float:
    """Fraction of pixels that are very bright (V >= v_min) and unsaturated (S <= s_max)."""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = (hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)
    return float(mask.mean())


def assess_global_quality(img_bgr: np.ndarray, cfg: ChromaProofConfig) -> Tuple[QualityMetrics, List[str]]:
    """Blur / exposure / glare checks. Returns (metrics, retake reasons)."""
    q = cfg.quality
    small, _ = downscale(img_bgr, cfg.analysis_max_side_px)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    metrics = QualityMetrics(
        image_width=int(img_bgr.shape[1]),
        image_height=int(img_bgr.shape[0]),
        blur_score=compute_blur_score(gray),
        blur_threshold=q.blur_threshold,
        mean_brightness=float(gray.mean()),
        contrast_std=float(gray.std()),
        glare_fraction_image=compute_glare_fraction(small, q.glare_v_min, q.glare_s_max),
    )

    reasons: List[str] = []
    # Blur is not judged on a too-dark image (Laplacian variance collapses with brightness).
    too_dark = metrics.mean_brightness < q.exposure_mean_min
    if metrics.blur_score < q.blur_threshold and not too_dark:
        reasons.append(
            f"Image is too blurry (blur score {metrics.blur_score:.1f} < threshold {q.blur_threshold:.1f})"
        )
    if metrics.mean_brightness < q.exposure_mean_min:
        reasons.append(
            f"Image exposure is unsuitable: too dark (mean brightness {metrics.mean_brightness:.0f} "
            f"< {q.exposure_mean_min:.0f})"
        )
    elif metrics.mean_brightness > q.exposure_mean_max:
        reasons.append(
            f"Image exposure is unsuitable: too bright (mean brightness {metrics.mean_brightness:.0f} "
            f"> {q.exposure_mean_max:.0f})"
        )
    elif metrics.contrast_std < q.exposure_min_contrast_std:
        reasons.append(
            f"Image exposure is unsuitable: contrast too low (std {metrics.contrast_std:.1f} "
            f"< {q.exposure_min_contrast_std:.1f})"
        )
    if metrics.glare_fraction_image > q.glare_max_fraction_image:
        reasons.append(
            f"Excessive glare detected ({metrics.glare_fraction_image:.1%} of image pixels "
            f"> {q.glare_max_fraction_image:.1%})"
        )
    return metrics, reasons


def card_glare_fraction(card_bgr: np.ndarray, cfg: ChromaProofConfig) -> float:
    """Glare fraction inside the rectified card (the area that matters for the result)."""
    rgb = card_bgr[..., ::-1]
    return float(glare_mask_rgb(rgb, cfg.quality.glare_v_min, cfg.quality.glare_s_max).mean())


# --------------------------------------------------------------------------- #
# Reference-card detection
# --------------------------------------------------------------------------- #
def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def _candidate_contours(gray: np.ndarray, cfg: ChromaProofConfig) -> list:
    c = cfg.card
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    contours: list = []
    # 1) strong edges (card border) -> closed outlines
    edges = cv2.Canny(blurred, c.canny_low, c.canny_high)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    found, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours.extend(found)
    # 2) dark border segmentation
    return contours


def _quad_from_contour(contour: np.ndarray, cfg: ChromaProofConfig) -> Optional[np.ndarray]:
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return None
    approx = cv2.approxPolyDP(hull, cfg.card.approx_epsilon_fraction * peri, True)
    if len(approx) != 4 or not cv2.isContourConvex(approx):
        return None
    return order_corners(approx.reshape(4, 2))


def _corner_cosines_ok(quad: np.ndarray, max_cos: float) -> bool:
    for i in range(4):
        a = quad[(i - 1) % 4] - quad[i]
        b = quad[(i + 1) % 4] - quad[i]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0 or abs(float(np.dot(a, b)) / denom) > max_cos:
            return False
    return True


def detect_card(img_bgr: np.ndarray, cfg: ChromaProofConfig) -> CardDetection:
    """Locate the supported reference card. Never guesses: returns found=False on doubt."""
    c = cfg.card
    small, scale = downscale(img_bgr, cfg.analysis_max_side_px)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    img_area = float(h * w)

    candidates = []
    for contour in _candidate_contours(gray, cfg):
        area = cv2.contourArea(cv2.convexHull(contour))
        frac = area / img_area
        if frac < c.min_area_fraction or frac > c.max_area_fraction:
            continue
        quad = _quad_from_contour(contour, cfg)
        if quad is None or not _corner_cosines_ok(quad, c.max_corner_cosine):
            continue
        width = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
        height = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
        if height <= 0:
            continue
        aspect = float(width / height)
        if abs(aspect / c.expected_aspect - 1.0) > c.aspect_tolerance:
            continue
        candidates.append((frac, aspect, quad))

    if not candidates:
        return CardDetection(found=False, reason="Reference card not visible (no card-shaped region found)")

    candidates.sort(key=lambda t: t[0], reverse=True)
    m = c.border_margin_px

    def touches_edge(quad: np.ndarray) -> bool:
        return bool(np.any(quad[:, 0] <= m) or np.any(quad[:, 0] >= w - 1 - m)
                    or np.any(quad[:, 1] <= m) or np.any(quad[:, 1] >= h - 1 - m))

    # The largest card-shaped candidate decides. If it touches the image edge the card is
    # cropped / partly outside the frame (a smaller inner rectangle must not be mistaken
    # for the whole card).
    frac, aspect, quad = candidates[0]
    if touches_edge(quad):
        return CardDetection(found=False, cropped=True,
                             reason="Reference card is cropped or partly outside the frame")
    return CardDetection(found=True, corners=(quad / scale).astype(np.float32),
                         area_fraction=float(frac), aspect_ratio=aspect)


def warp_card(img_bgr: np.ndarray, corners: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Perspective-rectify the card to ``size`` (width, height)."""
    w, h = size
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(np.asarray(corners, dtype=np.float32), dst)
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_LINEAR)
