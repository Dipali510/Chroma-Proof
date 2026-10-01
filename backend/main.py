import io
import json
import uuid
import sqlite3
import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
from datetime import datetime, timezone

import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from cryptography.hazmat.primitives.asymmetric import ed25519
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

app = FastAPI(title="ChromaProof API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CORE LOGIC FROM COLAB NOTEBOOK ---

class ChromaProofCrypto:
    def __init__(self):
        self._private_key = ed25519.Ed25519PrivateKey.generate()
        self._public_key = self._private_key.public_key()

    @property
    def public_key_hex(self) -> str:
        return self._public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw
        ).hex()

    @property
    def key_id(self) -> str:
        return f"KEY-ED25519-{self.public_key_hex[:8].upper()}"

    @staticmethod
    def sha256_digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    @staticmethod
    def canonicalize(record_dict: dict) -> str:
        subset = {
            "record_id": str(record_dict["record_id"]),
            "operator_id": str(record_dict["operator_id"]),
            "timestamp_utc": str(record_dict["timestamp_utc"]),
            "kit_id": str(record_dict["kit_id"]),
            "result": str(record_dict["result"]),
            "confidence": str(record_dict["confidence"]),
            "classifier_version": str(record_dict["classifier_version"]),
            "image_sha256": str(record_dict["image_sha256"]),
            "location": str(record_dict.get("location", "N/A"))
        }
        return json.dumps(subset, sort_keys=True, separators=(',', ':'), ensure_ascii=False)

    def sign(self, record_dict: dict) -> str:
        canonical_bytes = self.canonicalize(record_dict).encode("utf-8")
        return self._private_key.sign(canonical_bytes).hex()

    @classmethod
    def verify(cls, record_dict: dict, signature_hex: str, public_key_hex: str, image_bytes: bytes = None) -> Tuple[bool, str]:
        if image_bytes is not None:
            calc_hash = cls.sha256_digest(image_bytes)
            if calc_hash != record_dict.get("image_sha256"):
                return False, f"IMAGE TAMPER DETECTED: Computed ({calc_hash[:10]}...) != Stored"
        try:
            pub_key = ed25519.Ed25519PublicKey.from_public_bytes(bytes.fromhex(public_key_hex))
            canonical_bytes = cls.canonicalize(record_dict).encode("utf-8")
            pub_key.verify(bytes.fromhex(signature_hex), canonical_bytes)
            return True, "VERIFIED: Cryptographic Ed25519 signature valid."
        except InvalidSignature:
            return False, "TAMPER DETECTED: Digital signature mismatch."
        except Exception as ex:
            return False, f"VERIFICATION ERROR: {str(ex)}"

@dataclass
class KitProfile:
    kit_id: str
    display_name: str
    target_substance: str
    pos_centroid_lab: Tuple[float, float, float]
    neg_centroid_lab: Tuple[float, float, float]
    pos_threshold_delta_e: float
    neg_threshold_delta_e: float

class MultiKitRegistry:
    KITS = {
        "cobalt_thiocyanate": KitProfile("cobalt_thiocyanate", "Cobalt Thiocyanate (Scott Reagent)", "Cocaine HCl / Base", (36.0, 3.0, -48.0), (55.0, 45.0, 12.0), 32.0, 28.0),
        "marquis": KitProfile("marquis", "Marquis Reagent", "Opiates / MDMA", (21.0, 28.0, -18.0), (82.0, 0.0, 8.0), 30.0, 24.0),
        "duquenois_levine": KitProfile("duquenois_levine", "Duquenois-Levine Reagent", "Cannabinoids (THC)", (30.0, 31.0, -30.0), (75.0, 2.0, 22.0), 28.0, 24.0)
    }

class ChromaProofVision:
    VERSION = "ChromaProof-CV-v2.1-API"
    @classmethod
    def evaluate(cls, bgr_image: np.ndarray, kit_id: str) -> dict:
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        glare_ratio = float(np.sum(gray > 248) / gray.size)

        if lap_var < 35.0:
            return {"quality_status": "FAILED_QUALITY_GATE", "result": "INCONCLUSIVE", "confidence": "LOW (0.0%)", "explanation": "Image sharpness check failed.", "classifier_version": cls.VERSION}

        img_float = bgr_image.astype(np.float32)
        mean_b, mean_g, mean_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
        mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
        img_float[:, :, 0] *= (mean_gray / (mean_b + 1e-6))
        img_float[:, :, 1] *= (mean_gray / (mean_g + 1e-6))
        img_float[:, :, 2] *= (mean_gray / (mean_r + 1e-6))
        calibrated = np.clip(img_float, 0, 255).astype(np.uint8)

        cy, cx = h // 2, w // 2
        dy, dx = int(h * 0.15), int(w * 0.15)
        roi = calibrated[cy - dy : cy + dy, cx - dx : cx + dx]

        roi_lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
        measured_lab = np.array([float(np.mean(roi_lab[:, :, 0]) * (100.0 / 255.0)), float(np.mean(roi_lab[:, :, 1]) - 128.0), float(np.mean(roi_lab[:, :, 2]) - 128.0)])

        profile = MultiKitRegistry.KITS.get(kit_id, MultiKitRegistry.KITS["cobalt_thiocyanate"])
        delta_e_pos = float(np.linalg.norm(measured_lab - np.array(profile.pos_centroid_lab)))
        delta_e_neg = float(np.linalg.norm(measured_lab - np.array(profile.neg_centroid_lab)))

        if delta_e_pos <= profile.pos_threshold_delta_e and delta_e_pos < delta_e_neg:
            return {"quality_status": "PASSED", "result": "POSITIVE", "confidence": f"HIGH ({max(55.0, min(95.0, 100.0 - (delta_e_pos * 1.4))):.1f}%)", "explanation": "Reaction profile matches target.", "classifier_version": cls.VERSION}
        elif delta_e_neg <= profile.neg_threshold_delta_e and delta_e_neg < delta_e_pos:
            return {"quality_status": "PASSED", "result": "NEGATIVE", "confidence": f"HIGH ({max(50.0, min(92.0, 100.0 - (delta_e_neg * 1.4))):.1f}%)", "explanation": "No target shift detected.", "classifier_version": cls.VERSION}
        return {"quality_status": "PASSED", "result": "INCONCLUSIVE", "confidence": "UNCERTAIN (45.0%)", "explanation": "Ambiguous profile.", "classifier_version": cls.VERSION}

class ChromaProofDB:
    def __init__(self, db_path="chromaproof.db"):
        self.db_path = db_path
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS records (record_id TEXT PRIMARY KEY, operator_id TEXT, timestamp_utc TEXT, kit_id TEXT, result TEXT, confidence TEXT, classifier_version TEXT, explanation TEXT, image_sha256 TEXT, image_bytes BLOB, canonical_payload TEXT, signature_hex TEXT, key_id TEXT, public_key_hex TEXT, sync_status TEXT, location TEXT)")

    def save(self, rec: dict, img: bytes):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("INSERT INTO records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (rec["record_id"], rec["operator_id"], rec["timestamp_utc"], rec["kit_id"], rec["result"], rec["confidence"], rec["classifier_version"], rec["explanation"], rec["image_sha256"], img, rec["canonical_payload"], rec["signature_hex"], rec["key_id"], rec["public_key_hex"], rec["sync_status"], rec["location"]))

    def get(self, record_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            r = conn.execute("SELECT * FROM records WHERE record_id = ?", (record_id,)).fetchone()
            return dict(r) if r else None

# Runtime Instances
crypto_system = ChromaProofCrypto()
db_system = ChromaProofDB()

# --- FASTAPI ROUTES ---

@app.get("/api/kits")
def get_kits():
    return [{"id": k, "name": v.display_name} for k, v in MultiKitRegistry.KITS.items()]

@app.post("/api/process")
async def process_capture(image: UploadFile = File(...), operator_id: str = Form("OFFICER-4021"), kit_id: str = Form("cobalt_thiocyanate"), location: str = Form("GPS: Uploaded")):
    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    cv_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    cv_res = ChromaProofVision.evaluate(cv_bgr, kit_id)
    img_hash = crypto_system.sha256_digest(img_bytes)
    
    record_dict = {
        "record_id": f"CP-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}",
        "operator_id": operator_id, "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "kit_id": kit_id, "result": cv_res["result"], "confidence": cv_res["confidence"],
        "classifier_version": cv_res["classifier_version"], "explanation": cv_res["explanation"],
        "image_sha256": img_hash, "location": location
    }
    
    full_record = {**record_dict, "canonical_payload": crypto_system.canonicalize(record_dict), "signature_hex": crypto_system.sign(record_dict), "key_id": crypto_system.key_id, "public_key_hex": crypto_system.public_key_hex, "sync_status": "SYNCED_API"}
    db_system.save(full_record, img_bytes)
    return {"status": "SUCCESS", "record": full_record}

class VerifyReq(BaseModel):
    record_id: str
    simulate_tamper: bool = False

@app.post("/api/verify")
def verify_record(req: VerifyReq):
    rec = db_system.get(req.record_id)
    if not rec: return {"valid": False, "message": "Record not found in DB."}
    test_dict = {k: rec[k] for k in ["record_id", "operator_id", "timestamp_utc", "kit_id", "result", "confidence", "classifier_version", "image_sha256", "location"]}
    if req.simulate_tamper: test_dict["result"] = "NEGATIVE" if test_dict["result"] == "POSITIVE" else "POSITIVE"
    valid, msg = ChromaProofCrypto.verify(test_dict, rec["signature_hex"], rec["public_key_hex"], rec["image_bytes"])
    return {"valid": valid, "message": msg, "signer_key": rec["key_id"]}
