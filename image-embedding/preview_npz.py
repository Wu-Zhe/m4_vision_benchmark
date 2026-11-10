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

if __name__ == "__main__":
    preview_npz("embeddinggemma2_10_images_float16.npz", n=5)
