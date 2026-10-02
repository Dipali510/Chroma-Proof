"""ChromaProof Role 1 - Computer Vision & Classification (prototype/demo).

Public entry point for Role 2 / FastAPI::

    from vision import process_image
    response = process_image(image_bytes_or_ndarray_or_path)

NOT validated forensic/toxicology software. All results are presumptive.
(Names are imported lazily so ``python -m vision.processor`` runs without warnings.)
"""
from .config import DEFAULT_CONFIG, PROCESSING_VERSION, ChromaProofConfig

__all__ = ["process_image", "process_image_detailed", "DEFAULT_CONFIG", "ChromaProofConfig",
           "PROCESSING_VERSION"]


def __getattr__(name):
    if name in ("process_image", "process_image_detailed"):
        from . import processor
        return getattr(processor, name)
    raise AttributeError(f"module 'vision' has no attribute {name!r}")
