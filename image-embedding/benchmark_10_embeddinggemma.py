#!/usr/bin/env python3
"""Batch-embed 10 product images with the local EmbeddingGemma 2 checkpoint.

The output is a compressed NumPy archive containing the original image
filenames and float16, L2-normalized embeddings. The benchmark JSON separates
model-load, inference, serialization, and end-to-end timings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


DEFAULT_IMAGE_DIR = Path("/Users/zhewu/src/ecommerce-image-fetcher/output-bigfiles")
DEFAULT_MODEL_DIR = Path("/Users/zhewu/common-models/embeddinggemma-2")
ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "embeddinggemma2_10_images_float16.npz"
DEFAULT_REPORT = ROOT / "embeddinggemma2_10_images_benchmark.json"
DEFAULT_MANIFEST = ROOT / "embeddinggemma2_10_images.txt"


def cpu_seconds() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime + usage.ru_stime)


def device_info() -> dict:
    try:
        import torch

        return {
            "torch_version": torch.__version__,
            "mps_built": bool(torch.backends.mps.is_built()),
            "mps_available": bool(torch.backends.mps.is_available()),
        }
    except Exception as exc:  # pragma: no cover - only for diagnostics
        return {"torch_error": f"{type(exc).__name__}: {exc}"}


def gpu_utilization_note() -> dict:
    """Return honest telemetry status; macOS exposes no portable live GPU %."""
    info = device_info()
    if not info.get("mps_available", False):
        info.update({
            "gpu_utilization_percent": None,
            "gpu_utilization_status": "unavailable: PyTorch MPS is not available in this runtime",
        })
    else:
        info.update({
            "gpu_utilization_percent": None,
            "gpu_utilization_status": "not exposed by portable PyTorch/macOS API; MPS was used",
        })
    return info


def pick_images(image_dir: Path, count: int) -> list[Path]:
    images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".webp", ".jpg", ".jpeg", ".png"}
    )
    if len(images) < count:
        raise RuntimeError(f"Need {count} images, found {len(images)} in {image_dir}")
    return images[:count]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Local model not found: {args.model}")
    images = pick_images(args.image_dir, 10)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import SentenceTransformer

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device
    if device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device mps requested, but torch.backends.mps.is_available() is false")

    image_names = [p.name for p in images]
    image_inputs = [{"image": str(p)} for p in images]
    args.manifest.write_text("\n".join(image_names) + "\n", encoding="utf-8")

    process_start = time.perf_counter()
    cpu_start = cpu_seconds()
    load_start = time.perf_counter()
    model = SentenceTransformer(
        str(args.model),
        device=device,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": torch.float32},
    )
    load_seconds = time.perf_counter() - load_start

    # One batched encode keeps the vision tower resident and avoids ten separate
    # tokenizer/model dispatches. normalize_embeddings=True makes float16 storage
    # suitable for cosine search after a small dequantization to float32.
    encode_start = time.perf_counter()
    embeddings = np.asarray(
        model.encode(
            image_inputs,
            batch_size=args.batch_size,
            normalize_embeddings=True,
            truncate_dim=256,
            convert_to_numpy=True,
            show_progress_bar=False,
        ),
        dtype=np.float32,
    )
    encode_seconds = time.perf_counter() - encode_start
    if embeddings.shape != (10, 256):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    serialize_start = time.perf_counter()
    # Store only float16 vectors: 15,360 raw vector bytes for 10 x 256, then
    # ZIP compression. Filenames remain lossless UTF-8 NumPy strings.
    np.savez_compressed(
        args.output,
        filenames=np.asarray(image_names),
        embeddings=embeddings.astype(np.float16),
        embedding_dtype=np.asarray("float16"),
        normalized=np.asarray(True),
        model=np.asarray("google/embeddinggemma-2"),
    )
    serialize_seconds = time.perf_counter() - serialize_start
    end_to_end_seconds = time.perf_counter() - process_start
    cpu_used_seconds = cpu_seconds() - cpu_start
    cpu_percent_one_core = 100.0 * cpu_used_seconds / end_to_end_seconds
    cpu_percent_all_cores = cpu_percent_one_core / max(os.cpu_count() or 1, 1)

    report = {
        "model": "google/embeddinggemma-2",
        "model_path": str(args.model),
        "image_directory": str(args.image_dir),
        "filenames": image_names,
        "count": len(image_names),
        "embedding_shape": list(embeddings.shape),
        "stored_dtype": "float16",
        "stored_format": "npz (ZIP-compressed NumPy archive)",
        "device": device,
        "batch_size": args.batch_size,
        "timing_seconds": {
            "model_load": round(load_seconds, 6),
            "batched_encode_10_images": round(encode_seconds, 6),
            "serialize_npz": round(serialize_seconds, 6),
            "end_to_end_model_load_through_persist": round(end_to_end_seconds, 6),
            "per_image_encode": round(encode_seconds / 10.0, 6),
        },
        "cpu_utilization": {
            "process_cpu_seconds": round(cpu_used_seconds, 6),
            "process_percent_of_one_core_over_end_to_end": round(cpu_percent_one_core, 2),
            "process_percent_of_all_logical_cores_over_end_to_end": round(cpu_percent_all_cores, 2),
            "logical_cpu_count": os.cpu_count(),
        },
        "gpu": gpu_utilization_note(),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "outputs": {
            "embedding_archive": str(args.output),
            "image_manifest": str(args.manifest),
            "benchmark_report": str(args.report),
        },
        "method": "single batched SentenceTransformer.encode call; local checkpoint; no model download",
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
