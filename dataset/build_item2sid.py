# build_item2sid.py
import os
import pickle
import argparse
import numpy as np
import pandas as pd

# ---------- 数据读取（与你已有的一致） ----------
def parse_item_feature(item_feature_str):
    if pd.isna(item_feature_str) or not isinstance(item_feature_str, str):
        return np.array([])
    embeddings = []
    for emb_str in item_feature_str.split(";"):
        emb_str = emb_str.strip()
        if emb_str:
            emb = [float(x) for x in emb_str.split(",") if x.strip()]
            embeddings.append(emb)
    return np.array(embeddings, dtype=np.float32)

def read_recommendation_csv(file_path):
    df = pd.read_csv(file_path, sep='@', dtype=str)
    for col in ['timestamp', 'session_id', 'sequence_id', 'behavior_policy_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')
    if "item_feature" in df.columns:
        df["item_feature"] = df["item_feature"].apply(parse_item_feature)
    if "exposed_items" in df.columns:
        df["exposed_items"] = df["exposed_items"].apply(
            lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
        )
    return df

def build_global_item_dict(df):
    """
    返回: dict[item_id] = embedding(np.ndarray, shape=(d,))
    若同一 item 多次出现，保留最后一次（或你可以改成平均）。
    """
    item_dict = {}
    for _, row in df.iterrows():
        item_ids = row["exposed_items"]
        embeddings = row["item_feature"]
        if len(item_ids) != len(embeddings):
            # 行里 candidates 与 feature 对不上就跳过
            continue
        for iid, emb in zip(item_ids, embeddings):
            emb = np.asarray(emb, dtype=np.float32)
            item_dict[int(iid)] = emb
    return item_dict

# ---------- 量化（用 codebook 做逐级最近邻 + 残差） ----------
def load_codebooks(npz_path):
    """
    读取你之前保存的 embedding_codebook.npz
    返回: list[np.ndarray]，每级形状为 (V_l, d)
    """
    z = np.load(npz_path, allow_pickle=True)
    cbs = z["codebooks"]
    # 可能是 object 数组或真正的 3D/2D 数组
    if isinstance(cbs, np.ndarray) and cbs.dtype == object:
        codebooks = [np.asarray(cb, dtype=np.float32) for cb in cbs.tolist()]
    else:
        # 若保存成了 3D (L, V, d)，拆成 list
        if cbs.ndim == 3:
            codebooks = [cbs[i].astype(np.float32) for i in range(cbs.shape[0])]
        else:
            # 若已经是 list-like 也可直接转
            codebooks = [np.asarray(cb, dtype=np.float32) for cb in cbs]
    return codebooks

def encode_sid(emb, codebooks):
    """
    对单个向量 emb: (d,) 做 L 级量化：
      level l: 选最近中心 z_l = argmin_j ||residual - cb_l[j]||
               residual <- residual - cb_l[z_l]
    返回: tuple(z_1, ..., z_L)  （0-based 索引）
    """
    residual = emb.astype(np.float32).copy()
    tokens = []
    for cb in codebooks:
        # cb: (V_l, d)
        # 计算欧氏距离的 argmin
        # dist^2 = ||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
        # 用点积法更快
        xc = residual @ cb.T                     # (V_l,)
        x2 = (residual * residual).sum()         # scalar
        c2 = (cb * cb).sum(axis=1)               # (V_l,)
        d2 = x2 + c2 - 2.0 * xc
        z_l = int(np.argmin(d2))
        tokens.append(z_l)
        residual = residual - cb[z_l]
    return tuple(tokens)

def build_item2sid(item2vec, codebooks, verbose_every=50000):
    """
    item2vec: dict[item_id] = (d,)
    codebooks: list[np.ndarray]，每级 (V_l, d)
    返回: dict[item_id] = tuple(z1,...,zL)
    """
    item2sid = {}
    for k, (iid, emb) in enumerate(item2vec.items(), start=1):
        item2sid[int(iid)] = encode_sid(np.asarray(emb, dtype=np.float32), codebooks)
        if verbose_every and (k % verbose_every == 0):
            print(f"[build_item2sid] encoded {k} items...")
    return item2sid

def to_aligned_array(item2sid):
    """
    方便做快速索引的数组版表示：
      假设 item_id 从 1..N 连续（若不连续也能放在 max_id 长度的数组里）。
    返回:
      arr: np.ndarray, shape = (N+1, L), arr[i] 是 item i 的 SID（0行为占位）
           没有 SID 的位置填 -1
    """
    if not item2sid:
        return np.zeros((1, 0), dtype=np.int64)
    max_id = max(item2sid.keys())
    L = len(next(iter(item2sid.values())))
    arr = -np.ones((max_id + 1, L), dtype=np.int64)
    for iid, sid in item2sid.items():
        arr[int(iid)] = np.asarray(sid, dtype=np.int64)
    return arr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_path", type=str,default="rl4rs_dataset_a_rl_train.csv",
                    help="例如 rl4rs_dataset_a_rl_train.csv（或 _b_rl_train.csv），带 exposed_items 和 item_feature 列")
    ap.add_argument("--codebook_npz", type=str, default="embedding_codebook.npz",
                    help="由你的 SemanticIDGeneratorTorch 导出的 codebook 文件")
    ap.add_argument("--out_dir", type=str, default="./",
                    help="输出目录")
    ap.add_argument("--out_prefix", type=str, default="sid_index",
                    help="输出文件前缀：将写出 {out_prefix}_item2sid.pkl / .npy")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[1/4] 读取 CSV: {args.csv_path}")
    df = read_recommendation_csv(args.csv_path)
    print(f"  -> rows: {len(df)}")

    print(f"[2/4] 汇总 item embedding（exposed_items 对齐 item_feature）")
    item2vec = build_global_item_dict(df)
    print(f"  -> unique items: {len(item2vec)}")

    print(f"[3/4] 加载 codebooks: {args.codebook_npz}")
    codebooks = load_codebooks(args.codebook_npz)
    L = len(codebooks)
    print(f"  -> levels: {L} | vocab sizes: {[cb.shape[0] for cb in codebooks]} | dim: {codebooks[0].shape[1]}")

    print(f"[4/4] 编码 item → SID（逐级最近邻 + 残差量化）")
    item2sid = build_item2sid(item2vec, codebooks)
    arr = to_aligned_array(item2sid)

    pkl_path = os.path.join(args.out_dir, f"{args.out_prefix}_item2sid.pkl")
    npy_path = os.path.join(args.out_dir, f"{args.out_prefix}_item2sid.npy")
    with open(pkl_path, "wb") as f:
        pickle.dump(item2sid, f)
    np.save(npy_path, arr)

    print(f"保存完成：\n  - dict: {pkl_path}\n  - array: {npy_path}")
    # 小检验
    some_iid = next(iter(item2sid.keys()))
    print(f"示例: item {some_iid} -> SID {item2sid[some_iid]}")

if __name__ == "__main__":
    main()
