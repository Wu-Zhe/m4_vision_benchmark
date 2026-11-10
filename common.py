"""Paths and sample loading shared by the command-line utilities."""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
EMBEDDING_DIR = MODELS / "embeddinggemma-2"
QWEN_DIR = MODELS / "qwen3.5-9b-4bit"


def load_samples():
    manifest = SAMPLES / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing {manifest}. Run: python make_samples.py")
    samples = json.loads(manifest.read_text(encoding="utf-8"))
    if len(samples) != 5:
        raise ValueError(f"Expected 5 samples; got {len(samples)}")
    for sample in samples:
        sample["image_path"] = str((SAMPLES / sample["file"]).resolve())
        if not Path(sample["image_path"]).exists():
            raise FileNotFoundError(sample["image_path"])
    RESULTS.mkdir(parents=True, exist_ok=True)
    return samples
