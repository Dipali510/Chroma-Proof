"""Deterministic prototype classifier (no machine learning).

Rules, evaluated in order, on the calibrated median ROI colour in CIELAB using
CIE76 distances (dE76) to two configured demo reference colours:

  R1_OUTSIDE_RANGE : min(dE_pos, dE_neg) > outside_range_min_de  -> Inconclusive
  R2_POSITIVE      : dE_pos <= positive_max_de and dE_neg - dE_pos >= min_margin_de -> Positive
  R3_NEGATIVE      : dE_neg <= negative_max_de and dE_pos - dE_neg >= min_margin_de -> Negative
  R4_BORDERLINE    : anything else                                -> Inconclusive

Colours that fall outside the configured ranges are never forced into
Positive/Negative. Every output is PRESUMPTIVE only; the confidence value is a
heuristic indicator, not a probability of drug presence.
"""
from __future__ import annotations

from .config import ChromaProofConfig
from .models import (CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM, RESULT_INCONCLUSIVE,
                     RESULT_NEGATIVE, RESULT_POSITIVE, CalibrationResult, ClassificationOutcome,
                     RoiFeatures, delta_e76, rgb255_to_lab)


def _confidence(distance: float, margin: float, features: RoiFeatures,
                calibration: CalibrationResult, cfg: ChromaProofConfig) -> str:
    cf = cfg.confidence
    if distance <= cf.high_max_de and margin >= cf.high_min_margin_de:
        level = CONFIDENCE_HIGH
    elif distance <= cf.medium_max_de and margin >= cf.medium_min_margin_de:
        level = CONFIDENCE_MEDIUM
    else:
        level = CONFIDENCE_LOW
    if level == CONFIDENCE_HIGH and (calibration.mean_residual_de > cf.high_max_calibration_de
                                     or features.lab_std > cf.high_max_roi_std):
        level = CONFIDENCE_MEDIUM
    return level


def classify(features: RoiFeatures, calibration: CalibrationResult,
             cfg: ChromaProofConfig) -> ClassificationOutcome:
    cc = cfg.classification
    lab = features.lab_median
    d_pos = float(delta_e76(lab, rgb255_to_lab(cc.positive_reference_rgb)))
    d_neg = float(delta_e76(lab, rgb255_to_lab(cc.negative_reference_rgb)))
    nearest = min(d_pos, d_neg)
    shown = f"dE76 to positive reference {d_pos:.1f}, to negative reference {d_neg:.1f}"

    if nearest > cc.outside_range_min_de:
        return ClassificationOutcome(
            RESULT_INCONCLUSIVE, CONFIDENCE_LOW, "R1_OUTSIDE_RANGE",
            f"Rule R1_OUTSIDE_RANGE: colour is outside the configured demo ranges ({shown}; "
            f"nearest {nearest:.1f} > {cc.outside_range_min_de:.1f}) -> Inconclusive.", d_pos, d_neg)

    if d_pos <= cc.positive_max_de and (d_neg - d_pos) >= cc.min_margin_de:
        margin = d_neg - d_pos
        return ClassificationOutcome(
            RESULT_POSITIVE, _confidence(d_pos, margin, features, calibration, cfg), "R2_POSITIVE",
            f"Rule R2_POSITIVE: {shown}; positive distance <= {cc.positive_max_de:.1f} and margin "
            f"{margin:.1f} >= {cc.min_margin_de:.1f} -> Positive (presumptive).", d_pos, d_neg)

    if d_neg <= cc.negative_max_de and (d_pos - d_neg) >= cc.min_margin_de:
        margin = d_pos - d_neg
        return ClassificationOutcome(
            RESULT_NEGATIVE, _confidence(d_neg, margin, features, calibration, cfg), "R3_NEGATIVE",
            f"Rule R3_NEGATIVE: {shown}; negative distance <= {cc.negative_max_de:.1f} and margin "
            f"{margin:.1f} >= {cc.min_margin_de:.1f} -> Negative (presumptive).", d_pos, d_neg)

    return ClassificationOutcome(
        RESULT_INCONCLUSIVE, CONFIDENCE_LOW, "R4_BORDERLINE",
        f"Rule R4_BORDERLINE: colour is between the configured ranges or too close to both ({shown}) "
        f"-> Inconclusive.", d_pos, d_neg)
