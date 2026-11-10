#!/usr/bin/env python3
"""Batch-embed images listed in inputImageList.txt.

Binary format (little-endian):
    header: magic[8], version[H], dimension[H], dtype[H]
    record: filename_length[I], filename[UTF-8], embedding[256 x float16]
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import resource
import struct
import sys
import time
from pathlib import Path

import numpy as np

from benchmark_10_embeddinggemma import (
    GPU_MONITOR_SCRIPT,
    cpu_seconds,
    gpu_utilization_note,
    start_gpu_monitor,
    stop_gpu_monitor,
    summarize_gpu_usage,
)


MAGIC = b"EGIMG256"
VERSION = 2
DIMENSION = 256
DTYPE_FLOAT16 = 1
HEADER = struct.Struct("<8sHHH")
RECORD_LENGTH = struct.Struct("<I")
DEFAULT_MODEL = Path("/Users/zhewu/common-models/embeddinggemma-2")


def find_input_list(input_dir: Path) -> Path:
    """Find the input list, supporting plain text and gzip-compressed files."""
    plain_path = input_dir / "inputImageList.txt"
    gzip_path = input_dir / "inputImageList.txt.gz"
    if plain_path.is_file():
        return plain_path
    if gzip_path.is_file():
        return gzip_path
    raise FileNotFoundError(f"Missing image list: {plain_path} or {gzip_path}")


def read_image_list(
    image_dir: Path,
    input_dir: Path,
    validate_inputs: bool = False,
) -> tuple[list[str], list[Path]]:
    list_path = find_input_list(input_dir)

    names: list[str] = []
    paths: list[Path] = []
    if list_path.suffix == ".gz":
        list_stream = gzip.open(list_path, mode="rt", encoding="utf-8")
    else:
        list_stream = list_path.open("r", encoding="utf-8")

    with list_stream:
        lines = list_stream.read().splitlines()

    for raw_line in lines:
        name = raw_line.strip()
        if not name or name.startswith("#"):
            continue
        path = Path(name)
        resolved = path if path.is_absolute() else image_dir / path
        # Avoid one filesystem metadata lookup per image by default. Missing
        # files fail naturally during model.encode(); validation is opt-in for
        # large directories.
        if validate_inputs and not resolved.is_file():
            raise FileNotFoundError(f"Image listed in {list_path} not found: {resolved}")
        names.append(name)
        paths.append(resolved)

    if not names:
        raise ValueError(f"No image names found in {list_path}")
    return names, paths


def write_binary_header(output) -> int:
    header = HEADER.pack(MAGIC, VERSION, DIMENSION, DTYPE_FLOAT16)
    output.write(header)
    return len(header)


def write_binary_batch(output, names: list[str], embeddings: np.ndarray) -> int:
    if embeddings.shape != (len(names), DIMENSION):
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    bytes_written = 0
    for name, embedding in zip(names, embeddings, strict=True):
        filename = name.encode("utf-8")
        if len(filename) > 0xFFFFFFFF:
            raise ValueError(f"Filename is too long: {name}")
        output.write(RECORD_LENGTH.pack(len(filename)))
        output.write(filename)
        vector = np.asarray(embedding, dtype="<f2", order="C")
        output.write(vector.tobytes(order="C"))
        bytes_written += RECORD_LENGTH.size + len(filename) + vector.nbytes
    return bytes_written


def load_logged_names(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def append_unique_names(path: Path, names: list[str], known: set[str]) -> int:
    new_names = [name for name in names if name not in known]
    if not new_names:
        return 0
    with path.open("a", encoding="utf-8") as output:
        for name in new_names:
            output.write(name + "\n")
    known.update(new_names)
    return len(new_names)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the images; paths with spaces are supported",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Directory containing inputImageList.txt or inputImageList.txt.gz (default: image directory)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--start-record-number", type=int, default=1, help="1-based first input-list record to embed")
    parser.add_argument("--stop-record-number", type=int, help="1-based last input-list record to embed")
    parser.add_argument(
        "--validate-inputs",
        action="store_true",
        help="Check every listed image exists before embedding; slower for huge directories",
    )
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--output", type=Path, help="Binary output path; defaults to directory/image_embeddings.bin")
    # parser.add_argument("--report", type=Path, help="JSON report path; defaults to directory/image_embeddings_report.json")
    # parser.add_argument("--gpu-output", type=Path, help="powermetrics output; defaults to directory/gpu_usage_output.txt")
    args = parser.parse_args()

    print(args)

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    image_dir = args.directory.resolve()
    input_dir = (args.input or args.directory).resolve()
    output_path = (args.output / "image_embeddings.bin").resolve()
    report_path = (args.output / "image_embeddings_report.json").resolve()
    gpu_output_path = (args.output / "gpu_usage_output.txt").resolve()
    success_log_path = (args.output / "suc.txt").resolve()
    failure_log_path = (args.output / "failed.txt").resolve()
    # gpu_output_path = (args.output or image_dir / "gpu_usage_output.txt").resolve()
    print(f"output_path {output_path}")
    print(f"report_path {report_path}")
    print(f"gpu_output_path {gpu_output_path}")

    gpu_monitor = None
    gpu_monitor_file = None
    report: dict | None = None
    try:
        all_names, all_image_paths = read_image_list(
            image_dir,
            input_dir,
            validate_inputs=args.validate_inputs,
        )
        start_record_number = args.start_record_number
        stop_record_number = args.stop_record_number or len(all_names)
        if start_record_number < 1:
            raise ValueError("--start-record-number must be at least 1")
        if stop_record_number < start_record_number:
            raise ValueError("--stop-record-number must be >= --start-record-number")
        if start_record_number > len(all_names):
            raise ValueError(
                f"--start-record-number {start_record_number} exceeds input list size {len(all_names)}"
            )
        stop_record_number = min(stop_record_number, len(all_names))
        names = all_names[start_record_number - 1:stop_record_number]
        image_paths = all_image_paths[start_record_number - 1:stop_record_number]

        if not args.model.exists():
            raise FileNotFoundError(f"Local model not found: {args.model}")

        import torch
        from sentence_transformers import SentenceTransformer

        if args.device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            device = args.device
        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but torch.backends.mps.is_available() is false")

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

        # Start GPU telemetry only after model weights are loaded, so the GPU
        # report measures embedding inference and binary persistence rather
        # than model initialization.
        gpu_monitor, gpu_monitor_file = start_gpu_monitor(gpu_output_path)

        encode_seconds = 0.0
        write_seconds = 0.0
        bytes_written = 0
        batch_count = 0
        processed_count = 0
        attempted_count = 0
        failure_count = 0
        failed_images: list[str] = []
        success_log_path.parent.mkdir(parents=True, exist_ok=True)
        success_log_path.touch(exist_ok=True)
        failure_log_path.touch(exist_ok=True)
        logged_successes = load_logged_names(success_log_path)
        logged_failures = load_logged_names(failure_log_path)
        next_progress_report = 10
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as output:
            bytes_written += write_binary_header(output)
            for start in range(0, len(names), args.batch_size):
                batch_names = names[start:start + args.batch_size]
                batch_paths = image_paths[start:start + args.batch_size]
                attempted_count += len(batch_names)

                encode_start = time.perf_counter()
                try:
                    batch_embeddings = np.asarray(
                        model.encode(
                            [{"image": str(path)} for path in batch_paths],
                            batch_size=args.batch_size,
                            normalize_embeddings=True,
                            truncate_dim=DIMENSION,
                            convert_to_numpy=True,
                            show_progress_bar=False,
                        ),
                        dtype=np.float32,
                    )
                    if batch_embeddings.shape != (len(batch_names), DIMENSION):
                        raise RuntimeError(f"Unexpected embedding shape: {batch_embeddings.shape}")
                except Exception as exc:
                    encode_seconds += time.perf_counter() - encode_start
                    failure_count += len(batch_names)
                    failed_images.extend(batch_names)
                    append_unique_names(failure_log_path, batch_names, logged_failures)
                    print(
                        f"[batch failure] images {start + 1}-{start + len(batch_names)}: "
                        f"{type(exc).__name__}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                encode_seconds += time.perf_counter() - encode_start

                write_start = time.perf_counter()
                bytes_written += write_binary_batch(output, batch_names, batch_embeddings)
                output.flush()
                write_seconds += time.perf_counter() - write_start
                append_unique_names(success_log_path, batch_names, logged_successes)
                batch_count += 1
                processed_count += len(batch_names)

                if processed_count >= next_progress_report:
                    elapsed = time.perf_counter() - process_start
                    average = elapsed / processed_count
                    print(
                        f"[progress] elapsed={elapsed:.2f}s "
                        f"processed={processed_count}/{len(names)} "
                        f"attempted={attempted_count} failures={failure_count} "
                        f"average={average:.3f}s/image",
                        flush=True,
                    )
                    next_progress_report = ((processed_count // 10) + 1) * 10

        end_to_end_seconds = time.perf_counter() - process_start
        cpu_used_seconds = cpu_seconds() - cpu_start
        cpu_percent_one_core = 100.0 * cpu_used_seconds / end_to_end_seconds
        cpu_percent_all_cores = cpu_percent_one_core / max(os.cpu_count() or 1, 1)

        report = {
            "model": "google/embeddinggemma-2",
            "model_path": str(args.model),
            "image_directory": str(image_dir),
            "input_list": str(find_input_list(input_dir)),
            "input_record_count": len(all_names),
            "start_record_number": start_record_number,
            "stop_record_number": stop_record_number,
            "filenames": names,
            "count": len(names),
            "processed_count": processed_count,
            "attempted_count": attempted_count,
            "failure_count": failure_count,
            "failed_images": failed_images,
            "embedding_dimension": DIMENSION,
            "stored_dtype": "float16",
            "binary_format": {
                "magic": MAGIC.decode("ascii"),
                "version": VERSION,
                "header_bytes": HEADER.size,
                "record_layout": "uint32 filename_bytes + UTF-8 filename + 256 float16 values",
                "termination": "records continue until EOF",
                "bytes_written": bytes_written,
            },
            "device": device,
            "batch_size": args.batch_size,
            "batch_count": batch_count,
            "timing_seconds": {
                "model_load": round(load_seconds, 6),
                "batched_encode_all_images": round(encode_seconds, 6),
                "binary_write": round(write_seconds, 6),
                "end_to_end_model_load_through_persist": round(end_to_end_seconds, 6),
                "per_image_encode": round(encode_seconds / processed_count, 6) if processed_count else None,
            },
            "cpu_utilization": {
                "process_cpu_seconds": round(cpu_used_seconds, 6),
                "process_percent_of_one_core_over_end_to_end": round(cpu_percent_one_core, 2),
                "process_percent_of_all_logical_cores_over_end_to_end": round(cpu_percent_all_cores, 2),
                "logical_cpu_count": os.cpu_count(),
            },
            "gpu": {
                **gpu_utilization_note(device),
                "monitor_script": str(GPU_MONITOR_SCRIPT),
                "monitor_output": str(gpu_output_path),
                "monitor_pid": gpu_monitor.pid,
            },
            "host": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": sys.version,
            },
            "outputs": {
                "embedding_binary": str(output_path),
                "benchmark_report": str(report_path),
                "gpu_usage_output": str(gpu_output_path),
                "successful_images": str(success_log_path),
                "failed_images": str(failure_log_path),
            },
            "method": "one SentenceTransformer.encode call per batch; records written and flushed after each batch; local checkpoint; no model download",
        }
    finally:
        stop_gpu_monitor(gpu_monitor, gpu_monitor_file)
        if report is not None:
            powermetrics_summary = summarize_gpu_usage(gpu_output_path)
            active_residency = powermetrics_summary["gpu_active_residency_percent"]
            report["gpu"].update({
                "gpu_utilization_percent": active_residency["average"],
                "gpu_utilization_peak_percent": active_residency["peak"],
                "gpu_utilization_status": (
                    "GPU HW active residency parsed from powermetrics"
                    if active_residency["sample_count"]
                    else "no GPU active-residency samples parsed"
                ),
                "powermetrics_summary": powermetrics_summary,
            })
            report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
