import torch


class SemanticIDGeneratorTorch:
    """
    Residual quantization based semantic-id generator.
    """

    def __init__(self, n_levels=3, codebook_size=16, n_iter=50, device="cuda"):
        self.n_levels = n_levels
        self.codebook_size = codebook_size
        self.n_iter = n_iter
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.codebooks = []

    @torch.no_grad()
    def fit(self, item_embs):
        x = torch.as_tensor(item_embs, dtype=torch.float32, device=self.device)
        residual = x
        self.codebooks = []
        for _ in range(self.n_levels):
            codebook = self._kmeans(residual, self.codebook_size, self.n_iter)
            self.codebooks.append(codebook.detach().cpu())
            idx = self._nearest(residual, codebook)
            residual = residual - codebook[idx]
        return self

    @torch.no_grad()
    def encode(self, item_embs):
        x = torch.as_tensor(item_embs, dtype=torch.float32, device=self.device)
        residual = x
        codes = []
        for codebook in self.codebooks:
            cb = codebook.to(self.device)
            idx = self._nearest(residual, cb)
            codes.append(idx.detach().cpu())
            residual = residual - cb[idx]
        return torch.stack(codes, dim=1).numpy()

    @staticmethod
    def _nearest(x, codebook):
        dist = torch.cdist(x, codebook)
        return torch.argmin(dist, dim=1)

    def _kmeans(self, x, k, n_iter):
        n = x.size(0)
        perm = torch.randperm(n, device=x.device)[:k]
        centers = x[perm].clone()
        for _ in range(n_iter):
            idx = self._nearest(x, centers)
            new_centers = centers.clone()
            for j in range(k):
                mask = idx == j
                if mask.any():
                    new_centers[j] = x[mask].mean(dim=0)
            if torch.allclose(new_centers, centers):
                break
            centers = new_centers
        return centers
