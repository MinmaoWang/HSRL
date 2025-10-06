import numpy as np
from collections import Counter
from k_means_acc import SemanticIDGeneratorTorch
import pandas as pd
import os

    

def guess_dim(embedding_lines):
    dims = []
    for vec in embedding_lines[:10]:
        if isinstance(vec, (list, np.ndarray)):
            dims.append(len(vec))
    if not dims:
        return None
    return Counter(dims).most_common(1)[0][0]


### 1. 收集embedding到list
def load_embedding_file(file_path, max_samples=10000):
    embedding_lines = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            embedding_lines.append(line)
            if len(embedding_lines) >= max_samples:
                break
    return embedding_lines

### 2. 分析：格式异常、数值异常、重复、数值统计
def analyze_embedding_lines(embedding_lines, value_threshold=1.0):
    total = len(embedding_lines)
    value_list = []
    abnormal_lines = []
    outlier_lines = []
    repeat_lines = set()
    normal_lines = []
    line_set = set()

    for idx, vec in enumerate(embedding_lines):
        # 判重
        key = tuple(vec)
        if key in line_set:
            repeat_lines.add(key)
            abnormal_lines.append((idx, vec, '重复行'))
            continue
        line_set.add(key)

        try:
            floats = np.array(vec, dtype=np.float32)
        except Exception:
            abnormal_lines.append((idx, vec, '无法转float'))
            continue

        value_list.extend(floats.tolist())

        # 检查数值异常
        has_outlier = np.any(np.abs(floats) > value_threshold)
        if has_outlier:
            outlier_lines.append((idx, vec, '数值超出范围'))
        else:
            normal_lines.append(floats)

    unique = len(line_set)
    stats = {
        'count': len(value_list),
        'min': min(value_list) if value_list else 0,
        'max': max(value_list) if value_list else 0,
        'avg': np.mean(value_list) if value_list else 0
    }
    return {
        'total': total,
        'unique': unique,
        'abnormal': abnormal_lines,
        'outlier': outlier_lines,
        'normal_lines': normal_lines,
        'stats': stats
    }


def filter_valid_embeddings(embedding_lines, value_threshold=1.0, require_dim=None):
    embedding_set = set()
    embeddings = []
    for vec in embedding_lines:
        key = tuple(vec)
        if key in embedding_set:
            continue  # 去重

        # 维度检查
        if require_dim is not None and len(vec) != require_dim:
            continue

        # 数值范围检查
        floats = np.array(vec, dtype=np.float32)
        if not np.all((-value_threshold <= floats) & (floats <= value_threshold)):
            continue

        embeddings.append(floats)
        embedding_set.add(key)
    return np.array(embeddings, dtype=np.float32)


### 4. 再分析过滤后embedding
def analyze_embeddings(embeddings):
    flat = embeddings.flatten()
    stats = {
        'count': len(flat),
        'min': float(np.min(flat)) if len(flat) else 0,
        'max': float(np.max(flat)) if len(flat) else 0,
        'avg': float(np.mean(flat)) if len(flat) else 0,
    }
    return embeddings.shape[0], stats

### 打印详细分析报告
def print_analysis_report(analysis):
    print(f"总数: {analysis['total']}")
    print(f"去重后: {analysis['unique']} ({analysis['unique']/analysis['total']*100:.2f}%)")
    print(f"格式异常数: {len(analysis['abnormal'])}")
    print(f"数值超出[-1,1]的样本数: {len(analysis['outlier'])}")
    print(f"可用embedding样本数: {len(analysis['normal_lines'])}")
    print(f"数值统计: {analysis['stats']}")
    if analysis['abnormal']:
        print("\n--- 格式异常样本（前5条）---")
        for item in analysis['abnormal'][:5]:
            print(item)
    if analysis['outlier']:
        print("\n--- 数值异常样本（前5条）---")
        for item in analysis['outlier'][:5]:
            print(item)


def read_recommendation_csv(file_path):
    """
    读取推荐系统相关的CSV数据并进行基本处理
    
    参数:
        file_path: CSV文件路径
        
    返回:
        df: 处理后的DataFrame
    """
    try:
        
        # 读取CSV，注意这里假设字段间是用@分隔的
        df = pd.read_csv(
            file_path,
            sep='@',  # 分隔符为@
            # header=None,  # 无表头
            # names=columns,  # 指定列名
            dtype=str  # 先全部按字符串读取，避免数值解析错误
        )
        
        # 查看数据基本信息
        print(f"数据读取成功，共{df.shape[0]}行，{df.shape[1]}列")
        print("\n前5行数据预览：")
        print(df.head())
        
        # 可选：将部分字段转换为数值类型（根据实际需求处理）
        # 例如转换timestamp、session_id等为整数
        for col in ['item_feature']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='ignore')
        
        return df
    
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return None
    except Exception as e:
        print(f"读取文件时发生错误：{str(e)}")
        return None

def parse_item_feature(item_feature_str):
    """解析 item_feature 列为 embedding 矩阵"""
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
    """读取推荐系统 CSV 并解析"""
    df = pd.read_csv(file_path, sep='@', dtype=str)

    # 转换数值型字段
    for col in ['timestamp', 'session_id', 'sequence_id', 'behavior_policy_id']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='ignore')

    # 解析 embedding
    if "item_feature" in df.columns:
        df["item_feature"] = df["item_feature"].apply(parse_item_feature)

    # 解析 exposed_items (转为 list[int])
    if "exposed_items" in df.columns:
        df["exposed_items"] = df["exposed_items"].apply(
            lambda x: [int(i) for i in x.split(",")] if isinstance(x, str) else []
        )

    return df



def build_global_item_dict(df):
    item_dict = {}
    seen_count = 0
    unique_count = 0
    
    for _, row in df.iterrows():
        item_ids = row["exposed_items"]
        embeddings = row["item_feature"]
        
        # 防止 item_ids 和 embeddings 长度对不上
        if len(item_ids) != len(embeddings):
            print(f"[警告] 行数据长度不一致: items={len(item_ids)}, emb={len(embeddings)}")
            continue
        
        for i in range(len(item_ids)):
            item_id = item_ids[i]
            emb = embeddings[i]
            
            seen_count += 1
            if item_id not in item_dict:
                unique_count += 1
            
            item_dict[item_id] = emb  # 覆盖逻辑
    
    print(f"总共扫描到 {seen_count} 个 item 出现记录")
    print(f"去重后得到 {unique_count} 个唯一 item")
    print(f"最终字典大小: {len(item_dict)}")
    return item_dict


# 使用示例
if __name__ == "__main__":
    csv_file_path = "rl4rs_dataset_a_rl_train.csv"
    df = read_recommendation_csv(csv_file_path)

    # 构建全局 item embedding 字典
    global_item_dict = build_global_item_dict(df)
    
    print("全局字典大小:", len(global_item_dict))

    print(len(global_item_dict.values()))
    
    embeddings = list(global_item_dict.values())
    

    embedding_lines = embeddings

    # 收集 embedding 字符串到列表中
    print(f"初始样本数: {len(embedding_lines)}")
    
    
    # 自动推断最常见的 embedding 维度
    require_dim = guess_dim(embedding_lines)
    print(f"推断embedding维度为: {require_dim}")
    

    # 分析格式异常、重复、数值异常等
    analysis = analyze_embedding_lines(embedding_lines, value_threshold=100)
    print_analysis_report(analysis)
    
    # 过滤：只保留格式正确且数值在[-1,1]内且维度正确的 embedding
    embeddings = filter_valid_embeddings(embedding_lines, value_threshold=100, require_dim=require_dim)
    print(f"\n过滤后可用embedding: {embeddings.shape}")
    
    # 再次统计过滤后数据的数量与统计数值
    n_valid, stats2 = analyze_embeddings(embeddings)
    print(f"过滤后样本数: {n_valid}, 数值统计: {stats2}")
    
    # 训练语义ID编码器并保存 codebook
    generator = SemanticIDGeneratorTorch(n_levels=3, codebook_size=16, device="cuda")
    generator.set_verbose(True)
    generator.fit(embeddings)
    np.savez("embedding_codebook.npz", codebooks=[cb.numpy() for cb in generator.codebooks])
    # 打印绝对路径
    output_path = os.path.abspath("embedding_codebook.npz")
    print(f"Codebook 已保存，绝对路径: {output_path}")
    print("-----------训练完毕-----------")