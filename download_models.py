#!/usr/bin/env python3
"""Explicit download stage for the two HF models and the PaddleOCR v6 weights."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from common import EMBEDDING_DIR, QWEN_DIR, ROOT

EMBED_REPO = "google/embeddinggemma-2"
QWEN_REPO = "mlx-community/Qwen3.5-9B-4bit"


def download_hf(repo_id: str, destination: Path):
    from huggingface_hub import snapshot_download
    destination.mkdir(parents=True, exist_ok=True)
    print(f"\nDownloading {repo_id} -> {destination}", flush=True)
    snapshot_download(repo_id=repo_id, local_dir=str(destination))
    if not (destination / "config.json").exists():
        raise RuntimeError(f"Download seems incomplete: {destination}")
    print(f"Ready: {destination}")


def prepare_ocr():
    # Paddlex/PaddleOCR manages weights in its own official-model cache.
    # Instantiating the pipeline triggers the automatic download.
    from paddleocr import PaddleOCR
    from common import SAMPLES
    print("\nDownloading / caching PP-OCRv6 small detection and recognition models", flush=True)
    engine = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
    )
    # A first inference also checks that the downloaded models can initialize.
    test_file = SAMPLES / "red_cotton_tshirt.png"
    if test_file.exists():
        print("Performing OCR warm-up inference...", flush=True)
        next(iter(engine.predict(str(test_file))), None)
    print("OCR models cached. PaddleOCR manages their local path.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--embedding", action="store_true", help="Only EmbeddingGemma 2")
    group.add_argument("--qwen", action="store_true", help="Only Qwen3.5 9B 4-bit")
    group.add_argument("--ocr", action="store_true", help="Only PP-OCRv6")
    args = parser.parse_args()
    if args.ocr:
        prepare_ocr()
    elif args.embedding:
        download_hf(EMBED_REPO, EMBEDDING_DIR)
    elif args.qwen:
        download_hf(QWEN_REPO, QWEN_DIR)
    else:
        print("Downloading HF models. OCR is downloaded with the separate OCR environment.")
        download_hf(EMBED_REPO, EMBEDDING_DIR)
        download_hf(QWEN_REPO, QWEN_DIR)
        print("Then run .venv-ocr/bin/python download_models.py --ocr")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)
