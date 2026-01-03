import numpy as np
import torch 

def eu_dist2_torch(fea_a, fea_b=None, sqrt=True):
    """
    Euclidean distance matrix (PyTorch version)
    MATLAB: EuDist2

    Args:
        fea_a: Tensor of shape (n, d)
        fea_b: Tensor of shape (m, d) or None
        sqrt: whether to take sqrt

    Returns:
        Distance matrix:
            - (n, n) if fea_b is None
            - (n, m) otherwise
    """
    fea_a = fea_a.double()

    if fea_b is None:
        # aa: (n, 1)
        aa = torch.sum(fea_a * fea_a, dim=1, keepdim=True)
        # ab: (n, n)
        ab = fea_a @ fea_a.t()

        D = aa + aa.t() - 2 * ab
        D = torch.clamp(D, min=0)

        if sqrt:
            D = torch.sqrt(D)

        # 保证对称 & 对角为 0（对齐 MATLAB 行为）
        D = torch.maximum(D, D.t())
        D.fill_diagonal_(0)

        return D.abs()

    else:
        fea_b = fea_b.double()

        aa = torch.sum(fea_a * fea_a, dim=1, keepdim=True)  # (n, 1)
        bb = torch.sum(fea_b * fea_b, dim=1, keepdim=True)  # (m, 1)
        ab = fea_a @ fea_b.t()                              # (n, m)

        D = aa + bb.t() - 2 * ab
        D = torch.clamp(D, min=0)

        if sqrt:
            D = torch.sqrt(D)

        return D.abs()


def eu_dist2(fea_a, fea_b=None, sqrt=True):
    """
    Euclidean distance matrix
    MATLAB: EuDist2
    """
    fea_a = np.asarray(fea_a, dtype=np.float64)

    if fea_b is None:
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        ab = fea_a @ fea_a.T
        D = aa + aa.T - 2 * ab
        D = np.maximum(D, 0)

        if sqrt:
            D = np.sqrt(D)

        D = np.maximum(D, D.T)
        np.fill_diagonal(D, 0)
        return np.abs(D)

    else:
        fea_b = np.asarray(fea_b, dtype=np.float64)
        aa = np.sum(fea_a * fea_a, axis=1, keepdims=True)
        bb = np.sum(fea_b * fea_b, axis=1, keepdims=True)
        ab = fea_a @ fea_b.T
        D = aa + bb.T - 2 * ab
        D = np.maximum(D, 0)

        if sqrt:
            D = np.sqrt(D)

        return np.abs(D)