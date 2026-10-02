"""Central configuration for the ChromaProof vision prototype.

Every threshold, geometry value and rule used by the pipeline lives here.
All values are PROTOTYPE / DEMO values tuned on synthetic images only. They have
NOT been validated against real test kits, cameras or lighting conditions.

Override any value by building a modified copy::

    import dataclasses
    from vision.config import DEFAULT_CONFIG
    cfg = dataclasses.replace(
        DEFAULT_CONFIG,
        quality=dataclasses.replace(DEFAULT_CONFIG.quality, blur_threshold=60.0),
    )
    process_image(img, config=cfg)

DEMO REFERENCE-CARD LAYOUT (assumption, see README)
---------------------------------------------------
* Landscape rectangle, aspect ratio (width/height) ~1.5, thick near-black border.
* Photographed roughly upright (rotation < ~45 deg, perspective tilt allowed).
* After perspective-warping the card to ``card.warp_size`` all geometry below is
  expressed as fractions (0..1) of card width (x) and card height (y):
    - a row of 6 flat colour reference patches along the top,
    - one rectangular reagent window in the lower-middle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

PROCESSING_VERSION = "chromaproof-vision-0.1.0"

RGB = Tuple[int, int, int]


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class InputConfig:
    min_image_side_px: int = 300            # smaller images are rejected
    max_image_bytes: int = 25_000_000       # guard against huge uploads
    # An image with (almost) no coloured pixels is treated as greyscale/unsupported.
    colour_spread_min: int = 12             # max-min channel spread counted as "coloured"
    colour_pixel_fraction_min: float = 0.001


# --------------------------------------------------------------------------- #
# Quality gate
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QualityConfig:
    # Blur: variance of the Laplacian on the (down-scaled) grey image.
    # PROTOTYPE/DEMO threshold - must be re-tuned for real cameras.
    blur_threshold: float = 100.0
    # Exposure: mean grey level (0-255) and contrast (std of grey level).
    exposure_mean_min: float = 50.0
    exposure_mean_max: float = 215.0
    exposure_min_contrast_std: float = 12.0
    # Glare: pixels with HSV value >= glare_v_min AND saturation <= glare_s_max.
    glare_v_min: int = 250
    glare_s_max: int = 40
    glare_max_fraction_image: float = 0.02   # whole image
    glare_max_fraction_card: float = 0.01    # inside the detected card


# --------------------------------------------------------------------------- #
# Reference card
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatchSpec:
    name: str
    expected_rgb: RGB                  # nominal colour printed on the card (sRGB 0-255)
    center: Tuple[float, float]        # (x, y) fractions of the warped card
    half_size: Tuple[float, float]     # (half width, half height) fractions


def _default_patches() -> Tuple[PatchSpec, ...]:
    colours = (
        ("white", (235, 235, 235)),
        ("mid_grey", (128, 128, 128)),
        ("dark_grey", (55, 55, 55)),
        ("red", (200, 45, 45)),
        ("green", (45, 150, 60)),
        ("blue", (45, 70, 185)),
    )
    return tuple(
        PatchSpec(name, rgb, (0.1375 + 0.145 * i, 0.22), (0.055, 0.10))
        for i, (name, rgb) in enumerate(colours)
    )


@dataclass(frozen=True)
class CardConfig:
    # --- detection ---
    min_area_fraction: float = 0.05         # card area / image area
    max_area_fraction: float = 0.95
    expected_aspect: float = 1.5            # width / height
    aspect_tolerance: float = 0.20          # relative tolerance
    approx_epsilon_fraction: float = 0.02   # polygon approximation (x perimeter)
    max_corner_cosine: float = 0.5          # corners must be ~60..120 degrees
    canny_low: int = 50
    canny_high: int = 150
    dark_border_value_max: int = 80         # grey level treated as "dark border"
    border_margin_px: int = 3               # card corner this close to edge => cropped
    # --- geometry of the rectified card ---
    warp_size: Tuple[int, int] = (600, 400)  # (width, height) px
    patches: Tuple[PatchSpec, ...] = field(default_factory=_default_patches)


# --------------------------------------------------------------------------- #
# Colour calibration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CalibrationConfig:
    patch_sample_fraction: float = 0.5      # central fraction of each patch sampled
    patch_max_std: float = 20.0             # uniformity limit (mean per-channel std, 0-255)
    patch_max_glare_fraction: float = 0.2
    min_patches_required: int = 5           # of the configured patches
    min_tonal_range: float = 20.0           # observed patch values must span this range
    inlier_max_delta_e: float = 15.0        # residual limit for a patch to be kept
    max_mean_delta_e: float = 8.0           # mean residual limit after correction
    gain_range: Tuple[float, float] = (0.4, 2.5)
    offset_abs_max: float = 80.0


# --------------------------------------------------------------------------- #
# Reagent ROI
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RoiConfig:
    window: Tuple[float, float, float, float] = (0.28, 0.46, 0.72, 0.86)  # x0,y0,x1,y1
    inner_shrink: float = 0.6               # sample only the central 60% of the window
    min_pixels: int = 3000
    min_valid_fraction: float = 0.7         # share of ROI pixels not glare/shadow
    min_pixel_value: int = 25               # darker pixels are treated as shadow
    max_lab_std: float = 12.0               # uniformity limit (mean std of L,a,b)


# --------------------------------------------------------------------------- #
# Classification (deterministic demo rules, CIELAB distances, dE76)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClassificationConfig:
    # Demo reference colours for the (hypothetical) reagent outcomes - NOT validated.
    positive_reference_rgb: RGB = (110, 50, 140)    # violet
    negative_reference_rgb: RGB = (235, 225, 185)   # pale cream
    positive_max_de: float = 18.0           # R2: Positive needs dE to positive ref <= this
    negative_max_de: float = 18.0           # R3: Negative needs dE to negative ref <= this
    min_margin_de: float = 10.0             # chosen class must beat the other by this
    outside_range_min_de: float = 30.0      # R1: closer than this to at least one ref, else outside


@dataclass(frozen=True)
class ConfidenceConfig:
    """Heuristic uncertainty indicator. NOT a probability of drug presence."""
    high_max_de: float = 8.0
    high_min_margin_de: float = 30.0
    medium_max_de: float = 14.0
    medium_min_margin_de: float = 20.0
    high_max_calibration_de: float = 5.0    # worse calibration caps confidence at Medium
    high_max_roi_std: float = 6.0           # noisier ROI caps confidence at Medium


# --------------------------------------------------------------------------- #
# Top-level container
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChromaProofConfig:
    version: str = PROCESSING_VERSION
    analysis_max_side_px: int = 800         # images are down-scaled to this for analysis
    input: InputConfig = field(default_factory=InputConfig)
    quality: QualityConfig = field(default_factory=QualityConfig)
    card: CardConfig = field(default_factory=CardConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    classification: ClassificationConfig = field(default_factory=ClassificationConfig)
    confidence: ConfidenceConfig = field(default_factory=ConfidenceConfig)


DEFAULT_CONFIG = ChromaProofConfig()
