"""Data models / schema AND small colour-space helpers for the vision module.

The Role 2 contract is ``ProcessingResult.to_contract_dict()`` (six fields).
Everything else is optional debug information.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# --- allowed values (stable contract) ------------------------------------- #
QUALITY_PASS = "PASS"
QUALITY_RETAKE = "RETAKE"
QUALITY_ERROR = "ERROR"          # only for unexpected internal exceptions

RESULT_POSITIVE = "Positive"
RESULT_NEGATIVE = "Negative"
RESULT_INCONCLUSIVE = "Inconclusive"

CONFIDENCE_HIGH = "High"
CONFIDENCE_MEDIUM = "Medium"
CONFIDENCE_LOW = "Low"

CONTRACT_FIELDS = (
    "quality_status",
    "quality_reasons",
    "result",
    "confidence",
    "explanation",
    "version",
)


@dataclass
class QualityMetrics:
    image_width: int = 0
    image_height: int = 0
    blur_score: float = 0.0
    blur_threshold: float = 0.0
    mean_brightness: float = 0.0
    contrast_std: float = 0.0
    glare_fraction_image: float = 0.0
    glare_fraction_card: Optional[float] = None


@dataclass
class CardDetection:
    found: bool
    cropped: bool = False
    corners: Optional[np.ndarray] = None    # (4,2) TL,TR,BR,BL in original image pixels
    area_fraction: float = 0.0
    aspect_ratio: float = 0.0
    reason: Optional[str] = None


@dataclass
class PatchObservation:
    name: str
    expected_rgb: Tuple[float, float, float]
    observed_rgb: Tuple[float, float, float]
    std: float
    usable: bool                    # uniform, not glared, inside the card
    used: bool = False              # kept in the final calibration fit
    residual_de: Optional[float] = None
    note: str = ""


@dataclass
class CalibrationResult:
    success: bool
    gains: Optional[Tuple[float, float, float]] = None     # R, G, B
    offsets: Optional[Tuple[float, float, float]] = None   # R, G, B (0-255 scale)
    patches: List[PatchObservation] = field(default_factory=list)
    n_used: int = 0
    n_total: int = 0
    mean_residual_de: float = float("inf")
    reason: Optional[str] = None


@dataclass
class RoiFeatures:
    lab_median: Tuple[float, float, float]
    lab_mean: Tuple[float, float, float]
    lab_std: float                  # mean std of L, a, b
    hsv_of_median: Tuple[float, float, float]   # H deg, S 0-1, V 0-1
    n_pixels: int
    valid_fraction: float


@dataclass
class RoiResult:
    success: bool
    features: Optional[RoiFeatures] = None
    reason: Optional[str] = None


@dataclass
class ClassificationOutcome:
    result: str
    confidence: str
    rule_id: str
    rule_text: str
    delta_e_positive: float
    delta_e_negative: float


@dataclass
class ProcessingResult:
    quality_status: str
    quality_reasons: List[str]
    result: str
    confidence: str
    explanation: str
    version: str
    stage: str = ""                 # where processing stopped / finished (debug)
    metrics: Optional[QualityMetrics] = None
    card: Optional[CardDetection] = None
    calibration: Optional[CalibrationResult] = None
    roi: Optional[RoiResult] = None
    classification: Optional[ClassificationOutcome] = None

    def to_contract_dict(self) -> Dict[str, Any]:
        """The six agreed Role 1 -> Role 2 fields, JSON-serialisable."""
        return {
            "quality_status": self.quality_status,
            "quality_reasons": list(self.quality_reasons),
            "result": self.result,
            "confidence": self.confidence,
            "explanation": self.explanation,
            "version": self.version,
        }

    def to_debug_dict(self) -> Dict[str, Any]:
        """Contract fields plus all intermediate values (JSON-serialisable)."""
        out = self.to_contract_dict()
        out["debug"] = _jsonable(
            {
                "stage": self.stage,
                "metrics": self.metrics,
                "card": self.card,
                "calibration": self.calibration,
                "roi": self.roi,
                "classification": self.classification,
            }
        )
        return out


def _jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses / numpy values to plain JSON types."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return value if np.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# Colour-space helpers (sRGB 0-255 in; true CIELAB via float32 OpenCV conversion)
# --------------------------------------------------------------------------- #
def rgb255_to_lab(rgb) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.float32)
    shape = arr.shape
    flat = np.ascontiguousarray(np.clip(arr.reshape(-1, 1, 3) / 255.0, 0.0, 1.0))
    lab = cv2.cvtColor(flat, cv2.COLOR_RGB2LAB)
    return lab.reshape(shape)


def rgb255_to_hsv(rgb) -> np.ndarray:
    """Returns H in degrees (0-360), S and V in 0-1."""
    arr = np.asarray(rgb, dtype=np.float32)
    shape = arr.shape
    flat = np.ascontiguousarray(np.clip(arr.reshape(-1, 1, 3) / 255.0, 0.0, 1.0))
    hsv = cv2.cvtColor(flat, cv2.COLOR_RGB2HSV)
    return hsv.reshape(shape)


def delta_e76(lab_a, lab_b) -> np.ndarray:
    """CIE76 colour difference (Euclidean distance in CIELAB)."""
    diff = np.asarray(lab_a, dtype=np.float64) - np.asarray(lab_b, dtype=np.float64)
    return np.linalg.norm(diff, axis=-1)


def glare_mask_rgb(rgb255: np.ndarray, v_min: int, s_max: int) -> np.ndarray:
    """Glare = very bright (HSV value >= v_min) and nearly unsaturated (S <= s_max, 0-255).

    ``rgb255`` has shape (..., 3). Uses the same definition as the whole-image
    check (OpenCV 8-bit HSV: V = max channel, S = 255*(max-min)/max).
    """
    arr = np.asarray(rgb255, dtype=np.float32)
    vmax = arr.max(axis=-1)
    vmin = arr.min(axis=-1)
    sat = np.where(vmax > 0, 255.0 * (vmax - vmin) / np.maximum(vmax, 1e-6), 0.0)
    return (vmax >= v_min) & (sat <= s_max)
