"""Reference-card colour calibration AND reagent ROI extraction.

Method (deliberately simple and explainable):
1. Sample the median RGB of the central part of each reference patch.
2. Discard patches that are not uniform, glared, or inconsistent with the card.
3. Fit a per-channel linear correction  corrected = gain * observed + offset
   (least squares against the patches' nominal sRGB values).
4. Iteratively drop the worst patch (largest CIELAB dE76 residual) until every
   remaining patch is within ``inlier_max_delta_e``.
5. Fail explicitly if too few patches remain, the tonal range is too small, the
   gains are implausible or the mean residual is too large.

LIMITATIONS: this reduces, but does NOT eliminate, lighting/camera colour bias
(no gamma / white-balance / spectral modelling).
"""
from __future__ import annotations

from typing import List

import numpy as np

from .config import ChromaProofConfig
from .models import (CalibrationResult, PatchObservation, RoiFeatures, RoiResult, delta_e76,
                     glare_mask_rgb, rgb255_to_hsv, rgb255_to_lab)


def observe_patches(card_bgr: np.ndarray, cfg: ChromaProofConfig) -> List[PatchObservation]:
    """Measure every configured patch on the rectified card."""
    cc = cfg.calibration
    H, W = card_bgr.shape[:2]
    rgb = card_bgr[..., ::-1]
    observations: List[PatchObservation] = []
    for spec in cfg.card.patches:
        cx, cy = spec.center[0] * W, spec.center[1] * H
        hw = spec.half_size[0] * W * cc.patch_sample_fraction
        hh = spec.half_size[1] * H * cc.patch_sample_fraction
        x0, x1 = int(round(max(0, cx - hw))), int(round(min(W, cx + hw)))
        y0, y1 = int(round(max(0, cy - hh))), int(round(min(H, cy + hh)))
        crop = rgb[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
        if crop.shape[0] < 20:
            observations.append(PatchObservation(spec.name, tuple(map(float, spec.expected_rgb)),
                                                 (0.0, 0.0, 0.0), 0.0, False, note="patch outside card"))
            continue
        median = np.median(crop, axis=0)
        std = float(np.mean(np.std(crop, axis=0)))
        glare = float(glare_mask_rgb(crop, cfg.quality.glare_v_min, cfg.quality.glare_s_max).mean())
        note = ""
        usable = True
        if std > cc.patch_max_std:
            usable, note = False, f"not uniform (std {std:.1f} > {cc.patch_max_std:.1f})"
        elif glare > cc.patch_max_glare_fraction:
            usable, note = False, f"glare on patch ({glare:.0%})"
        observations.append(PatchObservation(
            spec.name, tuple(float(v) for v in spec.expected_rgb),
            tuple(float(v) for v in median), std, usable, note=note))
    return observations


def apply_correction(rgb255, calibration: CalibrationResult) -> np.ndarray:
    """Apply the fitted per-channel gain/offset to RGB values (0-255)."""
    if not calibration.success or calibration.gains is None or calibration.offsets is None:
        raise ValueError("Cannot apply a failed calibration")
    arr = np.asarray(rgb255, dtype=np.float32)
    return np.clip(arr * np.asarray(calibration.gains, np.float32)
                   + np.asarray(calibration.offsets, np.float32), 0.0, 255.0)


def _fit(observed: np.ndarray, expected: np.ndarray):
    gains, offsets = np.zeros(3), np.zeros(3)
    for ch in range(3):
        g, o = np.polyfit(observed[:, ch], expected[:, ch], 1)
        gains[ch], offsets[ch] = g, o
    return gains, offsets


def _residuals(observed: np.ndarray, expected: np.ndarray, gains, offsets) -> np.ndarray:
    corrected = np.clip(observed * gains + offsets, 0, 255)
    return delta_e76(rgb255_to_lab(corrected), rgb255_to_lab(expected))


def calibrate(card_bgr: np.ndarray, cfg: ChromaProofConfig) -> CalibrationResult:
    """Fit the colour correction from the card's reference patches."""
    cc = cfg.calibration
    patches = observe_patches(card_bgr, cfg)
    n_total = len(patches)

    def fail(reason: str) -> CalibrationResult:
        return CalibrationResult(success=False, patches=patches, n_total=n_total, reason=reason)

    active = [i for i, p in enumerate(patches) if p.usable]
    if len(active) < cc.min_patches_required:
        return fail(f"Reference card patches not sufficiently visible "
                    f"({len(active)} of {n_total} usable, need at least {cc.min_patches_required})")

    observed = np.array([patches[i].observed_rgb for i in range(n_total)], dtype=np.float64)
    expected = np.array([patches[i].expected_rgb for i in range(n_total)], dtype=np.float64)

    if np.any(np.ptp(observed[active], axis=0) < cc.min_tonal_range):
        return fail("Reference card patches show too little tonal range; card not recognised")

    # Iteratively drop the worst-fitting patch until all remaining are inliers.
    while True:
        gains, offsets = _fit(observed[active], expected[active])
        res = _residuals(observed[active], expected[active], gains, offsets)
        worst = int(np.argmax(res))
        if res[worst] <= cc.inlier_max_delta_e:
            break
        patches[active[worst]].note = f"rejected as outlier (dE76 {res[worst]:.1f})"
        patches[active[worst]].usable = False
        del active[worst]
        if len(active) < cc.min_patches_required:
            return fail("Reference card patches do not match the supported card layout "
                        f"(only {len(active)} of {n_total} consistent patches, need at least "
                        f"{cc.min_patches_required}); card may be unsupported, rotated or partly occluded")

    for k, i in enumerate(active):
        patches[i].used = True
        patches[i].residual_de = float(res[k])
    mean_res = float(np.mean(res))

    if np.any(gains < cc.gain_range[0]) or np.any(gains > cc.gain_range[1]):
        return fail(f"Colour calibration failed: implausible channel gains "
                    f"{np.round(gains, 2).tolist()} (allowed {cc.gain_range})")
    if np.any(np.abs(offsets) > cc.offset_abs_max):
        return fail(f"Colour calibration failed: implausible channel offsets {np.round(offsets, 1).tolist()}")
    if mean_res > cc.max_mean_delta_e:
        return fail(f"Colour calibration failed: mean residual dE76 {mean_res:.1f} "
                    f"> {cc.max_mean_delta_e:.1f}")

    return CalibrationResult(
        success=True,
        gains=tuple(float(g) for g in gains),
        offsets=tuple(float(o) for o in offsets),
        patches=patches, n_used=len(active), n_total=n_total, mean_residual_de=mean_res,
    )


# --------------------------------------------------------------------------- #
# Reagent ROI extraction and colour features
# (configured rectangle on the rectified card; glare/shadow excluded; calibrated;
#  summarised with robust medians in CIELAB / HSV)
# --------------------------------------------------------------------------- #
_PREFIX = "Reagent region cannot be identified"


def roi_bounds(card_shape, cfg: ChromaProofConfig):
    """Pixel bounds (x0, y0, x1, y1) of the sampled ROI on the rectified card."""
    H, W = card_shape[:2]
    x0f, y0f, x1f, y1f = cfg.roi.window
    cx, cy = 0.5 * (x0f + x1f) * W, 0.5 * (y0f + y1f) * H
    hw = 0.5 * (x1f - x0f) * W * cfg.roi.inner_shrink
    hh = 0.5 * (y1f - y0f) * H * cfg.roi.inner_shrink
    return (int(round(max(0, cx - hw))), int(round(max(0, cy - hh))),
            int(round(min(W, cx + hw))), int(round(min(H, cy + hh))))


def extract_roi(card_bgr: np.ndarray, calibration: CalibrationResult, cfg: ChromaProofConfig) -> RoiResult:
    rc = cfg.roi
    if not calibration.success:
        return RoiResult(False, reason=f"{_PREFIX}: calibration unavailable")
    x0, y0, x1, y1 = roi_bounds(card_bgr.shape, cfg)
    rgb = card_bgr[y0:y1, x0:x1, ::-1].reshape(-1, 3).astype(np.float32)
    if rgb.shape[0] < rc.min_pixels:
        return RoiResult(False, reason=f"{_PREFIX} (ROI too small: {rgb.shape[0]} px < {rc.min_pixels})")

    v = rgb.max(axis=1)
    glare = glare_mask_rgb(rgb, cfg.quality.glare_v_min, cfg.quality.glare_s_max)
    valid = ~glare & (v >= rc.min_pixel_value)
    valid_fraction = float(valid.mean())
    if valid_fraction < rc.min_valid_fraction or int(valid.sum()) < rc.min_pixels:
        return RoiResult(False, reason=f"{_PREFIX}: obscured by glare or shadow "
                                       f"({valid_fraction:.0%} usable pixels < {rc.min_valid_fraction:.0%})")

    corrected = apply_correction(rgb[valid], calibration)
    lab = rgb255_to_lab(corrected)
    lab_median = np.median(lab, axis=0)
    lab_std = float(np.mean(np.std(lab, axis=0)))
    if lab_std > rc.max_lab_std:
        return RoiResult(False, reason=f"{_PREFIX}: region is not uniform (colour spread {lab_std:.1f} "
                                       f"> {rc.max_lab_std:.1f})")

    hsv = rgb255_to_hsv(apply_correction(np.median(rgb[valid], axis=0), calibration))
    features = RoiFeatures(
        lab_median=tuple(float(x) for x in lab_median),
        lab_mean=tuple(float(x) for x in lab.mean(axis=0)),
        lab_std=lab_std,
        hsv_of_median=tuple(float(x) for x in hsv),
        n_pixels=int(valid.sum()),
        valid_fraction=valid_fraction,
    )
    return RoiResult(True, features=features)
