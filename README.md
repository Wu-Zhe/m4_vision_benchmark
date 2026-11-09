# Three-model image analysis smoke test on Apple M4 / 32 GB

**What this includes:** separate scripts for downloading and running **EmbeddingGemma 2**, **Qwen3.5-9B (4-bit)**, and **PP-OCRv6 small** against **five matching text queries and five synthetic product images**. Runs models sequentially to reduce unified-memory pressure.

## 1. Requirements

- Apple Silicon Mac, macOS 14+ recommended, ~15 GB free disk space for model weights, Python environments and outputs (allow additional headroom).
- **Native ARM64 Python 3.11 or 3.12**; not an Intel/Rosetta Python interpreter. Homebrew users can install `python@3.12` and set `PYTHON_BIN` accordingly.
- Internet connection for setup and initial downloads. Subsequent tests use local model paths and cached OCR weights.

## 2. Quick start

From Terminal in this project directory:

```bash
# Example only if python3 is 3.11 or 3.12:
bash setup_m4.sh
# Or: PYTHON_BIN=/opt/homebrew/bin/python3.12 bash setup_m4.sh

bash download_all.sh
bash run_all.sh
```

`setup_m4.sh` creates `.venv` for **EmbeddingGemma / Qwen / MLX** and `.venv-ocr` for **PaddleOCR / ONNX Runtime**. Two environments avoid PaddleOCR/PaddleX versus Transformers dependency conflicts. `download_all.sh` explicitly downloads both Hugging Face repositories to `models/` and initializes the two OCR models in the PaddleOCR-managed model cache.

If you want each part separately:

```bash
.venv/bin/python download_models.py --embedding
.venv/bin/python download_models.py --qwen
.venv-ocr/bin/python download_models.py --ocr

.venv/bin/python test_embeddings.py
.venv/bin/python test_qwen.py
.venv-ocr/bin/python test_ocr.py
```

Generate/re-generate sample images:

```bash
.venv/bin/python make_samples.py
```

## 3. Model IDs, runtimes, and output

| Task | Weights | Inference |
| --- | --- | --- |
| Text/image embeddings | `google/embeddinggemma-2` | `sentence-transformers` + PyTorch MPS; text+image tower (~440M effective params) |
| Image attributes | `mlx-community/Qwen3.5-9B-4bit` | `mlx-vlm`, MLX quantized 4-bit checkpoint |
| Image text/OCR | `PP-OCRv6_small_det` + `PP-OCRv6_small_rec` | `PaddleOCR`, ONNX Runtime CPU on Mac |

Generated files:

- `results/image_embeddings_768d.npy`: `(5, 768)` NumPy image vectors.
- `results/text_embeddings_768d.npy`: `(5, 768)` NumPy query vectors.
- `results/similarity_matrix.csv`: 5-by-5 query-to-image cosine similarities.
- `results/embeddings_report.json`: ranking, paired top-1 recall, timing.
- `results/qwen_attributes.json`: raw VLM generations and parsed product attributes.
- `results/ocr_results.json`: detected text strings and confidence scores.

**About the dataset:** `samples/manifest.json` pairs each PNG with a descriptive text query, expected color/category/size labels, and a reference printed string. The five illustrations are generated locally by Pillow so all tests are reproducible without external sample-photo URLs. These are *smoke tests*, not scientifically representative accuracy benchmarks. The images have deliberately legible product labels; consider swapping in real photographs after verifying that everything runs.

## 4. Switching to your own 5 images

Put five PNG/JPG files in `samples/` and update `samples/manifest.json` (keep exactly five entries). Required keys in each entry: `id`, `file`, `text_query`, `expected_attributes`, `expected_printed_text`.

Query documents with `prompt_name="SearchQuery"`; embed images as multimodal dictionaries like `{"image": "path/to/picture.png"}` without a text prompt. Both modalities produce comparable vectors, and the script normalizes them for cosine similarity.

## 5. Performance knobs

```bash
# 256 dimensions instead of 768; significantly smaller stored vectors
.venv/bin/python test_embeddings.py --dim 256

# Try bfloat16 on Apple Silicon (float32 is the stable default)
.venv/bin/python test_embeddings.py --dtype bfloat16

# Compare CPU vs MPS (the latter is the intended acceleration)
.venv/bin/python test_embeddings.py --device cpu

# Limit Qwen output tokens for lower latency
.venv/bin/python test_qwen.py --max-tokens 256
```

`Qwen3.5` normally enables reasoning/thinking by default. This test explicitly passes `enable_thinking=False` to the MLX chat template to reduce latency and request a direct JSON response. It also strips complete `<think>...</think>` sections if unexpectedly returned. If you see truncated JSON, increase `--max-tokens`. Unknown attributes should be `null`.

## 6. Common fixes

**Cannot install libraries:** Check `python3 --version` and `python3 -c 'import platform; print(platform.machine())'`. Use an Apple Silicon native Python 3.11/3.12. Update pip; try `rm -rf .venv .venv-ocr && bash setup_m4.sh` only if you are comfortable discarding the environment folders.

**MPS unsupported op:** Try `PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python test_embeddings.py`. If still unsuccessful, run `--device cpu` (slower). PyTorch MPS op coverage varies by installed version.

**Qwen not recognized or fails to load:** `pip install -U mlx-vlm` *inside* `.venv`. Ensure checkpoint is `mlx-community/Qwen3.5-9B-4bit`, not the text-only `mlx-lm` variant. The project uses the model card's `apply_chat_template` and `generate` API.

**PaddleOCR / ONNX package issue:** Confirm `.venv-ocr/bin/python -c 'import paddleocr, onnxruntime; print(onnxruntime.__version__)'`. PaddleOCR 3.x is required. First instantiation downloads OCR files to PaddleOCR's model cache, which is separate from `models/`.

**Running offline:** After first download, local model paths work offline. PaddleOCR may check its model cache or model source on initialization; ensure it has already warmed up successfully while online. For a completely offline deployment, pin dependencies and explicitly configure OCR model directories from the previously cached artifacts.

## 7. Verified sources

- EmbeddingGemma 2 model card: https://huggingface.co/google/embeddinggemma-2
- EmbeddingGemma 2 developer guide: https://developers.googleblog.com/embeddinggemma-2-the-developer-guide/
- Qwen3.5 9B Apple MLX checkpoint: https://huggingface.co/mlx-community/Qwen3.5-9B-4bit
- mlx-vlm usage: https://github.com/Neywa-Labs/mlx-vlm/blob/main/docs/usage.md
- Official PaddleOCR 3.x OCR usage: https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/OCR.en.md

**Verification note:** I checked the published model IDs and inference APIs, and locally validated the scripts/fixture generator. I could not execute the actual three model inferences in this environment (no Mac M4 and no model weight downloads), so device-specific installation and throughput are not represented as measured results.
# m4_vision_benchmark
# m4_vision_benchmark
