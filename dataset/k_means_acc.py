import torch
import numpy as np
from typing import List, Tuple

class SemanticIDGeneratorTorch:
    """语义ID生成器（torch加速版），多级平衡量化机制"""

    def __init__(self, n_levels: int = 3, codebook_size: int = 256,
                 balance_threshold: float = 0.05, device: str = 'cuda'):
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.balance_threshold = balance_threshold
        self.codebooks = [None for _ in range(n_levels)]
        self.verbose = False
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def set_verbose(self, verbose: bool):
        self.verbose = verbose

    def balanced_kmeans(self, data: torch.Tensor, k: int, max_iter: int = 100) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        data: [n_samples, n_features]
        return: centers, labels
        """
        n_samples = data.shape[0]
        if k > n_samples:
            raise ValueError("簇数量不能大于样本数量")

        indices = torch.randperm(n_samples)[:k]
        centers = data[indices].clone()
        labels = torch.zeros(n_samples, dtype=torch.long, device=data.device)

        target_size = n_samples // k
        remainder = n_samples % k

        for iter_idx in range(max_iter):
            # 计算距离矩阵 [n_samples, k]
            distances = torch.cdist(data, centers)

            # 当前簇大小
            cluster_sizes = torch.bincount(labels, minlength=k)

            # 平衡分配（for循环，不容易完全矢量化）
            for i in range(n_samples):
                nearest_indices = torch.argsort(distances[i])
                for idx in nearest_indices:
                    limit = target_size + (1 if idx.item() < remainder else 0)
                    if cluster_sizes[idx] < limit:
                        labels[i] = idx
                        cluster_sizes[idx] += 1
                        break

            # 重新计算质心
            new_centers = torch.zeros_like(centers)
            for c in range(k):
                mask = (labels == c)
                if mask.any():
                    new_centers[c] = data[mask].mean(dim=0)

            if torch.allclose(centers, new_centers, atol=1e-4):
                if self.verbose:
                    print(f"平衡K-means在第{iter_idx + 1}次迭代收敛")
                break

            centers = new_centers

        if self.verbose and iter_idx == max_iter - 1:
            print(f"平衡K-means达到最大迭代次数{max_iter}，未完全收敛")

        return centers, labels

    def fit(self, embeddings: np.ndarray):
        """
        embeddings: np.ndarray [n_samples, n_features]
        """
        # 转为torch
        emb = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
        n_samples, n_features = emb.shape
        residual = emb.clone()

        for level in range(self.n_levels):
            if self.verbose:
                print(f"训练第{level + 1}级代码本...")

            codebook, _ = self.balanced_kmeans(residual, self.codebook_size)
            self.codebooks[level] = codebook.detach().cpu()

            # 残差
            if level < self.n_levels - 1:
                dists = torch.cdist(residual, codebook)
                nearest_indices = dists.argmin(dim=1)
                codebook_selected = codebook[nearest_indices]
                residual = residual - codebook_selected

                if torch.norm(residual) < 1e-6:
                    if self.verbose:
                        print(f"第{level + 2}级残差接近零，提前终止训练")
                    break

        return self
