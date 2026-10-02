"""All ChromaProof Role 1 tests (synthetic demo images only).

Run from the project root:  python -m unittest discover -s tests -t .   (or: python -m pytest)
"""
import dataclasses
import os
import tempfile
import unittest
from functools import lru_cache
from unittest import mock

import cv2
import numpy as np

from vision import PROCESSING_VERSION, process_image, process_image_detailed
from vision import synthetic as S
from vision.calibration import apply_correction, calibrate
from vision.classifier import classify
from vision.config import DEFAULT_CONFIG
from vision.models import (CalibrationResult, RoiFeatures, delta_e76, rgb255_to_hsv, rgb255_to_lab)
from vision.quality import (assess_global_quality, card_glare_fraction, compute_blur_score,
                            detect_card, warp_card)


CONTRACT = {"quality_status", "quality_reasons", "result", "confidence", "explanation", "version"}


# --------------------------- shared synthetic fixtures ----------------------- #
@lru_cache(maxsize=None)
def scene(kind: str):
    """Return (image, homography) for a named synthetic scenario."""
    if kind == "positive":
        return S.render_scene(S.POSITIVE_RGB)
    if kind == "negative":
        return S.render_scene(S.NEGATIVE_RGB)
    if kind == "borderline":
        return S.render_scene(S.BORDERLINE_RGB)
    if kind == "unsupported":
        return S.render_scene(S.UNSUPPORTED_RGB)
    if kind == "missing_card":
        return S.render_scene(S.POSITIVE_RGB, with_card=False)
    if kind == "occluded":
        return S.render_scene(S.POSITIVE_RGB, occluded_patches=(0, 1))
    if kind == "cropped":
        return S.render_scene(S.POSITIVE_RGB, quad=[[-80, 100], [500, 115], [490, 470], [-90, 460]])
    if kind == "rotated":
        a = np.deg2rad(15)
        c, s = np.cos(a), np.sin(a)
        cx, cy = 380, 290
        quad = [[cx + c * (x - cx) - s * (y - cy), cy + s * (x - cx) + c * (y - cy)]
                for x, y in [[130, 120], [630, 130], [620, 450], [120, 440]]]
        return S.render_scene(S.POSITIVE_RGB, quad=quad)
    img, H = S.render_scene(S.POSITIVE_RGB)
    if kind == "blurry":
        return S.blur(img, 4.0), H
    if kind == "dark":
        return S.darken(img, 0.25), H
    if kind == "glare":
        return S.add_glare(img, S.roi_center_in_image(H), 90), H
    if kind == "glare_small":
        return S.add_glare(img, S.roi_center_in_image(H), 35), H
    raise KeyError(kind)


def image(kind: str):
    return scene(kind)[0]


def jpeg_bytes(kind: str, quality: int = 92) -> bytes:
    ok, buf = cv2.imencode(".jpg", image(kind), [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return buf.tobytes()



# --------------------------- Quality gate / card detection --------------------------- #

class QualityTests(unittest.TestCase):
    def test_good_image_passes_global_checks(self):
        metrics, reasons = assess_global_quality(image("positive"), DEFAULT_CONFIG)
        self.assertEqual(reasons, [])
        self.assertGreaterEqual(metrics.blur_score, DEFAULT_CONFIG.quality.blur_threshold)

    def test_blurry_image_is_flagged_with_score(self):
        metrics, reasons = assess_global_quality(image("blurry"), DEFAULT_CONFIG)
        self.assertTrue(any("too blurry" in r for r in reasons), reasons)
        self.assertLess(metrics.blur_score, DEFAULT_CONFIG.quality.blur_threshold)

    def test_blur_threshold_is_configurable(self):
        strict = dataclasses.replace(DEFAULT_CONFIG, quality=dataclasses.replace(
            DEFAULT_CONFIG.quality, blur_threshold=1e9))
        _, reasons = assess_global_quality(image("positive"), strict)
        self.assertTrue(any("too blurry" in r for r in reasons))

    def test_glare_is_detected(self):
        metrics, reasons = assess_global_quality(image("glare"), DEFAULT_CONFIG)
        self.assertTrue(any("Excessive glare detected" in r for r in reasons), reasons)
        self.assertGreater(metrics.glare_fraction_image, DEFAULT_CONFIG.quality.glare_max_fraction_image)

    def test_small_glare_on_card_is_detected_on_card(self):
        det = detect_card(image("glare_small"), DEFAULT_CONFIG)
        self.assertTrue(det.found)
        card = warp_card(image("glare_small"), det.corners, DEFAULT_CONFIG.card.warp_size)
        self.assertGreater(card_glare_fraction(card, DEFAULT_CONFIG), DEFAULT_CONFIG.quality.glare_max_fraction_card)

    def test_dark_image_exposure_flagged(self):
        _, reasons = assess_global_quality(image("dark"), DEFAULT_CONFIG)
        self.assertTrue(any("exposure is unsuitable" in r for r in reasons), reasons)

    def test_blur_score_orders_sharp_above_blurred(self):
        import cv2
        g_sharp = cv2.cvtColor(image("positive"), cv2.COLOR_BGR2GRAY)
        g_blur = cv2.cvtColor(image("blurry"), cv2.COLOR_BGR2GRAY)
        self.assertGreater(compute_blur_score(g_sharp), compute_blur_score(g_blur))

    def test_card_detection_variants(self):
        self.assertTrue(detect_card(image("positive"), DEFAULT_CONFIG).found)
        self.assertTrue(detect_card(image("rotated"), DEFAULT_CONFIG).found)   # 15 deg tilt
        self.assertFalse(detect_card(image("missing_card"), DEFAULT_CONFIG).found)
        cropped = detect_card(image("cropped"), DEFAULT_CONFIG)
        self.assertFalse(cropped.found)
        self.assertTrue(cropped.cropped)


# --------------------------- Calibration --------------------------- #

def _card(kind):
    img = image(kind)
    det = detect_card(img, DEFAULT_CONFIG)
    assert det.found, kind
    return warp_card(img, det.corners, DEFAULT_CONFIG.card.warp_size)


class CalibrationTests(unittest.TestCase):
    def test_good_card_calibrates_with_small_residual(self):
        cal = calibrate(_card("positive"), DEFAULT_CONFIG)
        self.assertTrue(cal.success, cal.reason)
        self.assertEqual(cal.n_used, cal.n_total)
        self.assertLess(cal.mean_residual_de, 3.0)

    def test_correction_undoes_colour_cast(self):
        cal = calibrate(_card("positive"), DEFAULT_CONFIG)
        raw_white = cal.patches[0].observed_rgb
        corrected = apply_correction(raw_white, cal)
        expected = np.array(DEFAULT_CONFIG.card.patches[0].expected_rgb)
        before = delta_e76(rgb255_to_lab(raw_white), rgb255_to_lab(expected))
        after = delta_e76(rgb255_to_lab(corrected), rgb255_to_lab(expected))
        self.assertLess(after, before)
        self.assertLess(after, 3.0)

    def test_occluded_patches_fail_explicitly(self):
        cal = calibrate(_card("occluded"), DEFAULT_CONFIG)
        self.assertFalse(cal.success)
        self.assertIn("patches", cal.reason)

    def test_blank_card_fails_instead_of_guessing(self):
        blank = np.full((400, 600, 3), 200, np.uint8)
        cal = calibrate(blank, DEFAULT_CONFIG)
        self.assertFalse(cal.success)

    def test_apply_correction_refuses_failed_calibration(self):
        cal = calibrate(np.full((400, 600, 3), 200, np.uint8), DEFAULT_CONFIG)
        with self.assertRaises(ValueError):
            apply_correction([10, 10, 10], cal)

    def test_missing_card_is_not_calibrated(self):
        self.assertFalse(detect_card(image("missing_card"), DEFAULT_CONFIG).found)


# --------------------------- Classifier --------------------------- #
CFG = DEFAULT_CONFIG.classification
GOOD_CAL = CalibrationResult(success=True, gains=(1, 1, 1), offsets=(0, 0, 0), n_used=6, n_total=6,
                             mean_residual_de=1.0)

def features_for(rgb, std=2.0):
    lab = tuple(float(x) for x in rgb255_to_lab(rgb))
    return RoiFeatures(lab, lab, std, tuple(float(x) for x in rgb255_to_hsv(rgb)), 5000, 1.0)


def blend(a, b, t):
    return tuple(int(round((1 - t) * x + t * y)) for x, y in zip(a, b))


class ClassifierTests(unittest.TestCase):
    def test_positive_reference_colour(self):
        out = classify(features_for(CFG.positive_reference_rgb), GOOD_CAL, DEFAULT_CONFIG)
        self.assertEqual((out.result, out.rule_id, out.confidence), ("Positive", "R2_POSITIVE", "High"))

    def test_negative_reference_colour(self):
        out = classify(features_for(CFG.negative_reference_rgb), GOOD_CAL, DEFAULT_CONFIG)
        self.assertEqual((out.result, out.rule_id), ("Negative", "R3_NEGATIVE"))

    def test_borderline_colour_is_inconclusive(self):
        rgb = blend(CFG.positive_reference_rgb, CFG.negative_reference_rgb, 0.25)
        out = classify(features_for(rgb), GOOD_CAL, DEFAULT_CONFIG)
        self.assertEqual(out.result, "Inconclusive")
        self.assertEqual(out.confidence, "Low")

    def test_unsupported_colour_is_inconclusive(self):
        out = classify(features_for((60, 170, 80)), GOOD_CAL, DEFAULT_CONFIG)
        self.assertEqual((out.result, out.rule_id), ("Inconclusive", "R1_OUTSIDE_RANGE"))

    def test_only_allowed_values_are_returned(self):
        for t in [i / 20 for i in range(21)]:
            rgb = blend(CFG.positive_reference_rgb, CFG.negative_reference_rgb, t)
            out = classify(features_for(rgb), GOOD_CAL, DEFAULT_CONFIG)
            self.assertIn(out.result, {"Positive", "Negative", "Inconclusive"})
            self.assertIn(out.confidence, {"High", "Medium", "Low"})

    def test_noisy_roi_or_poor_calibration_caps_confidence(self):
        noisy = classify(features_for(CFG.positive_reference_rgb, std=9.0), GOOD_CAL, DEFAULT_CONFIG)
        self.assertEqual(noisy.confidence, "Medium")
        poor = CalibrationResult(True, (1, 1, 1), (0, 0, 0), n_used=5, n_total=6, mean_residual_de=7.0)
        self.assertEqual(classify(features_for(CFG.positive_reference_rgb), poor, DEFAULT_CONFIG).confidence,
                         "Medium")

    def test_thresholds_come_from_config(self):
        rgb = blend(CFG.positive_reference_rgb, CFG.negative_reference_rgb, 0.25)
        loose = dataclasses.replace(DEFAULT_CONFIG, classification=dataclasses.replace(
            CFG, positive_max_de=30.0, outside_range_min_de=60.0))
        self.assertEqual(classify(features_for(rgb), GOOD_CAL, loose).result, "Positive")


# --------------------------- Processor (end to end) --------------------------- #

class ProcessorTests(unittest.TestCase):
    def assert_contract(self, out):
        self.assertEqual(set(out), CONTRACT)
        self.assertIn(out["quality_status"], {"PASS", "RETAKE", "ERROR"})
        self.assertIsInstance(out["quality_reasons"], list)
        self.assertIn(out["result"], {"Positive", "Negative", "Inconclusive"})
        self.assertIn(out["confidence"], {"High", "Medium", "Low"})
        self.assertTrue(out["explanation"])
        self.assertEqual(out["version"], PROCESSING_VERSION)

    # 1. good image
    def test_good_image(self):
        r = process_image_detailed(image("positive"))
        self.assertEqual(r.quality_status, "PASS")
        self.assertTrue(r.card.found)
        self.assertTrue(r.calibration.success)
        self.assertTrue(r.roi.success)
        self.assertEqual(r.result, "Positive")
        self.assertEqual(process_image_detailed(image("negative")).result, "Negative")

    # 2. blurry
    def test_blurry_image_rejected(self):
        out = process_image(image("blurry"))
        self.assert_contract(out)
        self.assertEqual(out["quality_status"], "RETAKE")
        self.assertTrue(any("blurry" in r for r in out["quality_reasons"]))
        self.assertEqual(out["result"], "Inconclusive")

    # 3. missing reference card
    def test_missing_card_does_not_continue_silently(self):
        r = process_image_detailed(image("missing_card"))
        self.assertEqual(r.quality_status, "RETAKE")
        self.assertIn("Reference card not visible", " ".join(r.quality_reasons))
        self.assertEqual(r.result, "Inconclusive")
        self.assertIsNone(r.calibration)
        self.assertIsNone(r.classification)

    # 4. glare
    def test_glare_image(self):
        for kind in ("glare", "glare_small"):
            out = process_image(image(kind))
            self.assertEqual(out["quality_status"], "RETAKE", kind)
            self.assertTrue(any("glare" in r for r in out["quality_reasons"]), kind)
            self.assertEqual(out["result"], "Inconclusive")

    # 5. borderline colour
    def test_borderline_colour_inconclusive(self):
        out = process_image(image("borderline"))
        self.assertEqual(out["quality_status"], "PASS")
        self.assertEqual(out["result"], "Inconclusive")
        self.assertIn("R4_BORDERLINE", out["explanation"])

    # 6. unsupported colour
    def test_unsupported_colour_inconclusive(self):
        out = process_image(image("unsupported"))
        self.assertEqual(out["result"], "Inconclusive")
        self.assertIn("R1_OUTSIDE_RANGE", out["explanation"])

    # 7. invalid / corrupted
    def test_invalid_inputs_produce_no_classification(self):
        good = jpeg_bytes("positive")
        bad_inputs = [None, b"", b"not an image", good[: len(good) // 3] + b"\x00" * 50,
                      np.zeros((0, 0, 3), np.uint8), np.zeros((500, 500), np.uint8),
                      np.zeros((500, 500, 3), np.uint16), np.zeros((50, 50, 3), np.uint8),
                      "/no/such/file.png", 12345, np.full((500, 500, 3), 128, np.uint8)]
        for bad in bad_inputs:
            r = process_image_detailed(bad)
            self.assert_contract(r.to_contract_dict())
            self.assertNotEqual(r.quality_status, "PASS")
            self.assertEqual(r.result, "Inconclusive")
            self.assertIsNone(r.classification)

    # 8. contract on a normal valid demo image (array, bytes and path)
    def test_contract_fields_for_all_input_types(self):
        self.assert_contract(process_image(image("positive")))
        out_bytes = process_image(jpeg_bytes("positive"))
        self.assert_contract(out_bytes)
        self.assertEqual(out_bytes["result"], "Positive")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "kit.png")
            cv2.imwrite(path, image("negative"))
            out = process_image(path)
        self.assert_contract(out)
        self.assertEqual(out["result"], "Negative")

    # extras
    def test_occluded_and_cropped_card_are_retakes(self):
        for kind in ("occluded", "cropped", "dark"):
            out = process_image(image(kind))
            self.assertEqual(out["quality_status"], "RETAKE", kind)
            self.assertEqual(out["result"], "Inconclusive", kind)

    def test_rotated_card_still_classified(self):
        self.assertEqual(process_image(image("rotated"))["result"], "Positive")

    def test_internal_exception_never_fakes_a_result(self):
        with mock.patch("vision.processor.detect_card", side_effect=RuntimeError("boom")):
            out = process_image(image("positive"))
        self.assert_contract(out)
        self.assertEqual((out["quality_status"], out["result"]), ("ERROR", "Inconclusive"))

    def test_explanation_never_claims_confirmation(self):
        text = process_image(image("positive"))["explanation"].lower()
        self.assertIn("presumptive", text)
        self.assertIn("laboratory confirmation is required", text)

    def test_generated_sample_data(self):
        expected = {"good/positive_demo.jpg": ("PASS", "Positive"), "good/negative_demo.jpg": ("PASS", "Negative"),
                    "blurry/blurry_demo.jpg": ("RETAKE", "Inconclusive"),
                    "glare/glare_demo.jpg": ("RETAKE", "Inconclusive"),
                    "missing_card/missing_card_demo.jpg": ("RETAKE", "Inconclusive"),
                    "borderline/borderline_demo.jpg": ("PASS", "Inconclusive"),
                    "borderline/unsupported_colour_demo.jpg": ("PASS", "Inconclusive")}
        with tempfile.TemporaryDirectory() as d:
            outputs = S.generate_samples(d)
            self.assertTrue(os.path.isfile(os.path.join(d, "example_outputs.json")))
        for key, (status, result) in expected.items():
            self.assert_contract(outputs[key])
            self.assertEqual((outputs[key]["quality_status"], outputs[key]["result"]), (status, result), key)


if __name__ == "__main__":
    unittest.main()
