import numpy as np

def preview_npz(path, n=5):
    # allow_pickle=True is only needed if the file contains object arrays;
    # only enable it for files you trust.
    with np.load(path, allow_pickle=False) as data:
        print(f"Keys in file: {data.files}\n")
        for key in data.files:
            arr = data[key]
            print(f"--- {key} --- shape={arr.shape}, dtype={arr.dtype}")
            if arr.ndim == 0:
                print(arr.item())          # scalar
            else:
                print(arr[:n])             # first n records
            print()


def preview_pairs(path, n=3, show_values=8):
    with np.load(path, allow_pickle=False) as data:
        filenames = data["filenames"]
        embeddings = data["embeddings"].astype(np.float32)  # float16 -> float32 for cleaner printing

        # Metadata (0-d arrays, so use .item())
        print(f"model       : {data['model'].item()}")
        print(f"dtype       : {data['embedding_dtype'].item()}")
        print(f"normalized  : {data['normalized'].item()}")
        print(f"total items : {len(filenames)}, embedding dim: {embeddings.shape[1]}\n")

        for i, (name, vec) in enumerate(zip(filenames[:n], embeddings[:n]), start=1):
            print(f"[{i}] {name}")
            print(f"    first {show_values} values: {np.round(vec[:show_values], 4)}")
            print(f"    L2 norm: {np.linalg.norm(vec):.4f}")
            print()


if __name__ == "__main__":
    preview_npz("embeddinggemma2_10_images_float16.npz", n=5)
    preview_pairs("embeddinggemma2_10_images_float16.npz", n=3)
