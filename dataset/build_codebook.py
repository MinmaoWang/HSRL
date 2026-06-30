import argparse
import os
import pickle

import numpy as np

from k_means_acc import SemanticIDGeneratorTorch


def load_item_embeddings(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        ids = np.asarray(list(obj.keys()), dtype=np.int64)
        embs = np.asarray(list(obj.values()), dtype=np.float32)
        return ids, embs
    arr = np.asarray(obj, dtype=np.float32)
    ids = np.arange(arr.shape[0], dtype=np.int64)
    return ids, arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item2vec", required=True, help="item embedding pickle")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--out_prefix", default="sid_index")
    parser.add_argument("--n_levels", type=int, default=3)
    parser.add_argument("--codebook_size", type=int, default=16)
    parser.add_argument("--n_iter", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    ids, embs = load_item_embeddings(args.item2vec)
    generator = SemanticIDGeneratorTorch(
        n_levels=args.n_levels,
        codebook_size=args.codebook_size,
        n_iter=args.n_iter,
        device=args.device,
    )
    generator.fit(embs)
    out = {
        "item_ids": ids,
        "codebooks": [cb.numpy() if hasattr(cb, "numpy") else cb for cb in generator.codebooks],
        "n_levels": args.n_levels,
        "codebook_size": args.codebook_size,
    }
    path = os.path.join(args.out_dir, f"{args.out_prefix}_codebook.pkl")
    with open(path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
