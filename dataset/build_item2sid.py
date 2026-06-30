# build_item2sid.py
import argparse
import os
import pickle

import numpy as np


def load_codebooks(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict) and "codebooks" in obj:
        return obj["codebooks"]
    return obj


def encode_sid(emb, codebooks):
    residual = np.asarray(emb, dtype=np.float32)
    sid = []
    for cb in codebooks:
        cb = np.asarray(cb, dtype=np.float32)
        dist = ((cb - residual[None, :]) ** 2).sum(axis=1)
        idx = int(np.argmin(dist))
        sid.append(idx)
        residual = residual - cb[idx]
    return sid


def load_item2vec(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def build_item2sid(item2vec, codebooks, verbose_every=50000):
    item2sid = {}
    for k, (iid, emb) in enumerate(item2vec.items(), start=1):
        item2sid[int(iid)] = encode_sid(np.asarray(emb, dtype=np.float32), codebooks)
        if verbose_every and k % verbose_every == 0:
            print(f"[build_item2sid] encoded {k} items...")
    return item2sid


def to_aligned_array(item2sid):
    if not item2sid:
        return np.zeros((1, 0), dtype=np.int64)
    max_id = max(item2sid.keys())
    L = len(next(iter(item2sid.values())))
    arr = np.full((max_id + 1, L), -1, dtype=np.int64)
    for iid, sid in item2sid.items():
        arr[int(iid), :] = np.asarray(sid, dtype=np.int64)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--item2vec", required=True, help="item embedding pickle")
    parser.add_argument("--codebook", required=True, help="codebook pickle")
    parser.add_argument("--out_dir", default=".")
    parser.add_argument("--out_prefix", default="sid_index")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    codebooks = load_codebooks(args.codebook)
    item2vec = load_item2vec(args.item2vec)
    print("[4/4] 编码 item → SID（逐级最近邻 + 残差量化）")
    item2sid = build_item2sid(item2vec, codebooks)
    arr = to_aligned_array(item2sid)

    pkl_path = os.path.join(args.out_dir, f"{args.out_prefix}_item2sid.pkl")
    npy_path = os.path.join(args.out_dir, f"{args.out_prefix}_item2sid.npy")
    with open(pkl_path, "wb") as f:
        pickle.dump(item2sid, f)
    np.save(npy_path, arr)
    some_iid = next(iter(item2sid.keys()))
    print(f"示例: item {some_iid} -> SID {item2sid[some_iid]}")
    print(f"Saved: {pkl_path}")
    print(f"Saved: {npy_path}")


if __name__ == "__main__":
    main()
