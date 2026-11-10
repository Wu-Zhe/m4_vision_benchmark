#!/usr/bin/env python3
"""Run PP-OCRv6 small detection + recognition against the same five images."""
from __future__ import annotations
import json
import sys
import time
from collections.abc import Mapping
from common import RESULTS, load_samples


def result_dict(r):
    """Support PaddleOCR/PaddleX result objects with .json or Mapping APIs."""
    data = getattr(r, "json", None)
    if callable(data):
        data = data()
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            data = {}
    if isinstance(data, Mapping):
        d = dict(data)
    elif isinstance(r, Mapping):
        d = dict(r)
    else:
        try:
            d = {"rec_texts": r["rec_texts"], "rec_scores": r["rec_scores"]}
        except (KeyError, TypeError):
            d = {}
    nested = d.get("res")
    return nested if isinstance(nested, Mapping) else d


def main():
    from paddleocr import PaddleOCR
    samples = load_samples()
    print("Initializing PP-OCRv6 small models (ONNX Runtime / CPU)...", flush=True)
    t0 = time.perf_counter()
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_small_det",
        text_recognition_model_name="PP-OCRv6_small_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        engine="onnxruntime",
    )
    load_s = time.perf_counter() - t0
    outputs = []
    for sample in samples:
        t0 = time.perf_counter()
        results = list(ocr.predict(sample["image_path"]))
        seconds = time.perf_counter() - t0
        texts, scores = [], []
        for r in results:
            data = result_dict(r)
            texts.extend([str(x) for x in data.get("rec_texts", [])])
            scores.extend([round(float(x), 4) for x in data.get("rec_scores", [])])
        result = {"image": sample["file"], "expected_example_text": sample["expected_printed_text"],
                  "elapsed_seconds": round(seconds, 2), "recognized_text": texts,
                  "confidence_scores": scores}
        outputs.append(result)
        print(f"{sample['file']} ({seconds:.2f}s): {texts}")
    report = {"models": ["PP-OCRv6_small_det", "PP-OCRv6_small_rec"],
              "engine": "onnxruntime", "load_seconds": round(load_s, 2), "samples": outputs}
    (RESULTS / "ocr_results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {RESULTS / 'ocr_results.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"OCR test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
