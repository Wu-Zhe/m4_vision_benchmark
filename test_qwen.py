#!/usr/bin/env python3
"""Run Qwen3.5-9B 4-bit on five product images and save extracted JSON."""
from __future__ import annotations
import argparse
import json
import re
import sys
import time
from pathlib import Path
from common import QWEN_DIR, RESULTS, load_samples

PROMPT = """Analyze the provided product image ONLY. Extract attributes visible in the image.
Return one valid JSON object with EXACTLY these fields:
{
  "product_type": string or null,
  "primary_color": string or null,
  "secondary_colors": list of strings,
  "material": string or null,
  "brand_or_printed_logo": string or null,
  "size_or_capacity": string or null,
  "model_number": string or null,
  "visible_text": list of strings
}
Don't infer physical size without a printed size, scale or measurement.
Use null if uncertain. Do not invent brand names or materials. No commentary, no markdown.
"""


def parse_json(raw: str):
    # Some thinking-capable models emit a reasoning section before the final answer.
    without_thinking = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    without_fences = re.sub(r"```(?:json)?\s*|\s*```", "", without_thinking).strip()
    try:
        return json.loads(without_fences)
    except (json.JSONDecodeError, ValueError):
        pass
    # Best effort: extract the outermost JSON object from residual model prose.
    decoder = json.JSONDecoder()
    for index, ch in enumerate(without_fences):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(without_fences[index:])
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue
    return None


def output_text(value):
    if isinstance(value, str):
        return value
    text = getattr(value, "text", None)
    return text if isinstance(text, str) else str(value)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=str(QWEN_DIR))
    p.add_argument("--max-tokens", type=int, default=384)
    args = p.parse_args()
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} missing. Run: .venv/bin/python download_models.py --qwen")
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    samples = load_samples()
    print(f"Loading MLX model: {model_path}", flush=True)
    t0 = time.perf_counter()
    model, processor = load(str(model_path))
    config = load_config(str(model_path))
    load_s = time.perf_counter() - t0
    formatted_prompt = apply_chat_template(
        processor, config, PROMPT, num_images=1, enable_thinking=False
    )

    outputs = []
    for sample in samples:
        print(f"Analyzing {sample['file']}...", flush=True)
        t0 = time.perf_counter()
        generation = generate(
            model, processor, formatted_prompt,
            image=[sample["image_path"]],
            max_tokens=args.max_tokens, temperature=0.0, verbose=False,
        )
        seconds = time.perf_counter() - t0
        raw = output_text(generation)
        structured = parse_json(raw)
        result = {
            "file": sample["file"], "expected_attributes": sample["expected_attributes"],
            "elapsed_seconds": round(seconds, 2), "valid_json": isinstance(structured, dict),
            "extracted": structured, "raw_output": raw,
        }
        outputs.append(result)
        print(json.dumps({"file": sample["file"], "time_s": round(seconds, 2),
                          "valid_json": result["valid_json"], "extracted": structured},
                         ensure_ascii=False))

    report = {"model": str(model_path), "load_seconds": round(load_s, 2),
              "total_inference_seconds": round(sum(x["elapsed_seconds"] for x in outputs), 2),
              "samples": outputs, "note": "Expected labels are for manual comparison, not input to Qwen."}
    (RESULTS / "qwen_attributes.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved {RESULTS / 'qwen_attributes.json'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Qwen test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
