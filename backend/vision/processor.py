"""Main integration entry point for Role 2 (FastAPI).

    from vision import process_image
    out = process_image(image)      # dict with exactly the six contract fields

``image`` may be: raw encoded bytes (e.g. an uploaded JPEG/PNG), a NumPy array
(8-bit BGR/BGRA as returned by ``cv2.imread``), or a file path (str / Path).

The function NEVER raises and NEVER fabricates a Positive/Negative result when
processing fails: every failure returns ``result = "Inconclusive"`` with a clear
reason. ``quality_status`` is "PASS" or "RETAKE"; "ERROR" is used only for an
unexpected internal exception.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .calibration import calibrate, extract_roi
from .classifier import classify
from .config import DEFAULT_CONFIG, ChromaProofConfig
from .models import (CONFIDENCE_LOW, QUALITY_ERROR, QUALITY_PASS, QUALITY_RETAKE, RESULT_INCONCLUSIVE,
                     CalibrationResult, CardDetection, ProcessingResult, QualityMetrics, RoiResult)
from .quality import assess_global_quality, card_glare_fraction, detect_card, warp_card

_NOTICE = ("This is a presumptive prototype result only; laboratory confirmation is required.")


# --------------------------------------------------------------------------- #
# Input loading / validation
# --------------------------------------------------------------------------- #
def load_image(image: Any, cfg: ChromaProofConfig) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Return (BGR uint8 image, None) or (None, reason)."""
    ic = cfg.input
    if image is None:
        return None, "Invalid image: no image supplied"

    if isinstance(image, (str, os.PathLike)):
        path = os.fspath(image)
        if not os.path.isfile(path):
            return None, "Invalid image: file not found"
        if os.path.getsize(path) > ic.max_image_bytes:
            return None, "Invalid image: file too large"
        with open(path, "rb") as fh:
            image = fh.read()

    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
        if len(data) == 0:
            return None, "Empty image: no data supplied"
        if len(data) > ic.max_image_bytes:
            return None, "Invalid image: data too large"
        decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            return None, "Invalid image: corrupted or unsupported image format"
        image = decoded

    if not isinstance(image, np.ndarray):
        return None, f"Unsupported image input type: {type(image).__name__}"
    if image.size == 0:
        return None, "Empty image: array has no pixels"
    if image.dtype != np.uint8:
        return None, f"Unsupported image condition: expected 8-bit pixels, got {image.dtype}"
    if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
        return None, "Unsupported image condition: greyscale image (colour image required)"
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        return None, f"Unsupported image condition: unexpected array shape {image.shape}"
    if image.shape[2] == 4:
        image = image[..., :3]
    if min(image.shape[:2]) < ic.min_image_side_px:
        return None, (f"Image resolution too low ({image.shape[1]}x{image.shape[0]} px; "
                      f"shortest side must be >= {ic.min_image_side_px} px)")
    img = np.ascontiguousarray(image)
    spread = img.max(axis=2).astype(np.int16) - img.min(axis=2).astype(np.int16)
    if float((spread >= ic.colour_spread_min).mean()) < ic.colour_pixel_fraction_min:
        return None, "Unsupported image condition: image appears to be greyscale (no colour information)"
    return img, None


# --------------------------------------------------------------------------- #
# Result builders
# --------------------------------------------------------------------------- #
def _retake(cfg: ChromaProofConfig, reasons: List[str], stage: str, **parts) -> ProcessingResult:
    explanation = (
        "Image rejected by the quality gate; no classification was attempted. "
        "Problems found: " + "; ".join(reasons) + ". Please retake the photo. "
        "Result is reported as Inconclusive. " + _NOTICE
    )
    return ProcessingResult(QUALITY_RETAKE, reasons, RESULT_INCONCLUSIVE, CONFIDENCE_LOW,
                            explanation, cfg.version, stage=stage, **parts)


def _error(cfg: ChromaProofConfig, exc: Exception) -> ProcessingResult:
    reason = f"Processing error: {type(exc).__name__}: {exc}"
    return ProcessingResult(
        QUALITY_ERROR, [reason], RESULT_INCONCLUSIVE, CONFIDENCE_LOW,
        "An unexpected error occurred while processing the image; no classification was produced. "
        "Result is reported as Inconclusive. " + _NOTICE,
        cfg.version, stage="error")


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _run(image: Any, cfg: ChromaProofConfig) -> ProcessingResult:
    img, load_reason = load_image(image, cfg)
    if img is None:
        return _retake(cfg, [load_reason or "Invalid image"], stage="input")

    metrics, reasons = assess_global_quality(img, cfg)
    glare_flagged = any(r.startswith("Excessive glare") for r in reasons)

    card = detect_card(img, cfg)
    calibration: Optional[CalibrationResult] = None
    roi: Optional[RoiResult] = None
    stage = "quality"

    if not card.found:
        reasons.append(card.reason or "Reference card not visible")
        stage = "card_detection"
    else:
        card_img = warp_card(img, card.corners, cfg.card.warp_size)
        metrics.glare_fraction_card = card_glare_fraction(card_img, cfg)
        if metrics.glare_fraction_card > cfg.quality.glare_max_fraction_card:
            if not glare_flagged:
                reasons.append(
                    f"Excessive glare detected on the reference card "
                    f"({metrics.glare_fraction_card:.1%} of card pixels > "
                    f"{cfg.quality.glare_max_fraction_card:.1%})")
            stage = "card_glare"
        else:
            calibration = calibrate(card_img, cfg)
            if not calibration.success:
                reasons.append(calibration.reason or "Colour calibration failed")
                stage = "calibration"
            else:
                roi = extract_roi(card_img, calibration, cfg)
                if not roi.success:
                    reasons.append(roi.reason or "Reagent region cannot be identified")
                    stage = "roi"

    if reasons:
        return _retake(cfg, reasons, stage, metrics=metrics, card=card, calibration=calibration, roi=roi)

    outcome = classify(roi.features, calibration, cfg)
    f = roi.features
    explanation = " ".join([
        f"Quality gate passed (blur score {metrics.blur_score:.0f} >= {metrics.blur_threshold:.0f}; "
        f"mean brightness {metrics.mean_brightness:.0f}; glare {metrics.glare_fraction_image:.1%} of image).",
        f"Reference card detected; {calibration.n_used}/{calibration.n_total} patches used for per-channel "
        f"gain/offset colour calibration (mean residual dE76 {calibration.mean_residual_de:.1f}).",
        f"Reagent region median colour after calibration: L*={f.lab_median[0]:.1f}, a*={f.lab_median[1]:.1f}, "
        f"b*={f.lab_median[2]:.1f} (hue {f.hsv_of_median[0]:.0f} deg; {f.n_pixels} px; colour spread "
        f"{f.lab_std:.1f}).",
        outcome.rule_text,
        f"Confidence '{outcome.confidence}' is a heuristic indicator, not a probability. " + _NOTICE,
    ])
    return ProcessingResult(QUALITY_PASS, [], outcome.result, outcome.confidence, explanation, cfg.version,
                            stage="classified", metrics=metrics, card=card, calibration=calibration,
                            roi=roi, classification=outcome)


def process_image_detailed(image: Any, config: Optional[ChromaProofConfig] = None) -> ProcessingResult:
    """Full result object including intermediate values. Never raises."""
    cfg = config or DEFAULT_CONFIG
    try:
        return _run(image, cfg)
    except Exception as exc:  # noqa: BLE001 - safety net: never fake a result
        return _error(cfg, exc)


def process_image(image: Any, config: Optional[ChromaProofConfig] = None) -> Dict[str, Any]:
    """Role 2 contract. Returns exactly:

        {"quality_status", "quality_reasons", "result", "confidence", "explanation", "version"}
    """
    return process_image_detailed(image, config).to_contract_dict()


# --------------------------------------------------------------------------- #
# Command line:  python -m vision.processor path/to/image.jpg [--debug]
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    import json
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = [a for a in argv if not a.startswith("--")]
    if len(paths) != 1:
        print("usage: python -m vision.processor <image> [--debug]", file=sys.stderr)
        return 2
    out = (process_image_detailed(paths[0]).to_debug_dict() if "--debug" in argv
           else process_image(paths[0]))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
