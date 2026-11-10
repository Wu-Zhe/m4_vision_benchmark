#!/usr/bin/env python3
"""Embed five text queries and five images, then rank image hits per query."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
from common import EMBEDDING_DIR, RESULTS, load_samples


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--dim", type=int, choices=[768, 512, 256, 128], default=768)
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--model", default=str(EMBEDDING_DIR))
    args = p.parse_args()

    import torch
    from sentence_transformers import SentenceTransformer
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("WARNING: MPS unavailable; falling back to CPU.")
        args.device = "cpu"
    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} is missing. Run: .venv/bin/python download_models.py --embedding")

    samples = load_samples()
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    print(f"Loading EmbeddingGemma 2 | {args.device} | {args.dtype} | {args.dim} dims", flush=True)
    start = time.perf_counter()
    # Vision is required, but audio is unnecessary; effective model ~440M params.
    model = SentenceTransformer(
        str(model_path),
        device=args.device,
        config_kwargs={"audio_config": None},
        model_kwargs={"torch_dtype": dtype},
    )
    load_seconds = time.perf_counter() - start
    image_data = [{"image": sample["image_path"]} for sample in samples]
    queries = [sample["text_query"] for sample in samples]

    t0 = time.perf_counter()
    image_emb = np.asarray(model.encode(
        image_data, batch_size=1, normalize_embeddings=True,
        truncate_dim=args.dim, convert_to_numpy=True, show_progress_bar=True,
    ), dtype=np.float32)
    image_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    text_emb = np.asarray(model.encode(
        queries, batch_size=2, normalize_embeddings=True,
        prompt_name="SearchQuery", truncate_dim=args.dim,
        convert_to_numpy=True, show_progress_bar=True,
    ), dtype=np.float32)
    text_seconds = time.perf_counter() - t0

    if image_emb.shape != (5, args.dim) or text_emb.shape != (5, args.dim):
        raise RuntimeError(f"Unexpected embedding shapes: images={image_emb.shape}, text={text_emb.shape}")
    scores = text_emb @ image_emb.T  # 5 queries x 5 images; normalized cosine similarities
    top1 = np.argmax(scores, axis=1)
    recall1 = float(np.mean(top1 == np.arange(5)))
    np.save(RESULTS / f"image_embeddings_{args.dim}d.npy", image_emb)
    np.save(RESULTS / f"text_embeddings_{args.dim}d.npy", text_emb)

    with (RESULTS / "similarity_matrix.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["query \\ image"] + [s["file"] for s in samples])
        for i, sample in enumerate(samples):
            writer.writerow([sample["text_query"]] + [f"{float(v):.5f}" for v in scores[i]])

    ranking = []
    for i, query in enumerate(queries):
        matches = [{"image": samples[j]["file"], "score": round(float(scores[i, j]), 5)}
                   for j in np.argsort(-scores[i]).tolist()]
        ranking.append({"query": query, "expected_image": samples[i]["file"], "matches": matches})
        print(f"\nQuery {i+1}: {query}")
        print(f"  #1 {matches[0]['image']}, similarity={matches[0]['score']:.4f}"
              f" {'(correct)' if top1[i] == i else '(not the paired image)'}")

    report = {
        "model": str(model_path), "device": args.device, "dtype": args.dtype,
        "embedding_dimension": args.dim, "image_shape": list(image_emb.shape),
        "text_shape": list(text_emb.shape), "top1_paired_recall": recall1,
        "timing_seconds": {"load": round(load_seconds, 2), "five_images": round(image_seconds, 2),
                           "five_texts": round(text_seconds, 2)},
        "rankings": ranking,
        "caveat": "Images are synthetic smoke-test fixtures, not a representative retrieval benchmark.",
    }
    (RESULTS / "embeddings_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nPaired image top-1: {recall1:.0%}. Saved vectors, similarity_matrix.csv, embeddings_report.json")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Embedding test failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
