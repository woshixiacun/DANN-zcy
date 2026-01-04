import numpy as np
import scipy.sparse as sp
from .distance_calculate import eu_dist2
from .distance_calculate import eu_dist2_torch
from .adjacency import adjacency
from .adjacency import adjacency_torch
import torch

def m_locaglob_torch(DATA, TYPE='nn', PARAM=6, device=None):
    """
    PyTorch version of locaglob.m

    Returns:
        L  : local Laplacian (numpy array)
        Ln : global Laplacian (numpy array)
    """
    if not torch.is_tensor(DATA):
        DATA = torch.tensor(DATA, dtype=torch.float32)

    if device is not None:
        DATA = DATA.to(device)

    # DATA = DATA.double() # 8*800
    n = DATA.shape[0]  #8

    print('\nLaplacian Eigenmaps Embedding.')

    # -------- adjacency --------
    A = adjacency_torch(DATA, TYPE=TYPE, PARAM=PARAM, device=device)
    W = A.copy().tolil()

    # non-zero entries
    A_i, A_j = A.nonzero()

    # -------- beta --------
    # squared distance, no sqrt
    D_full = eu_dist2_torch(DATA, sqrt=False)
    beta = torch.mean(torch.sum(D_full, dim=1)).item()

    # -------- Local weight matrix W --------
    for i, j in zip(A_i, A_j):
        # squared distance between i and j
        DD = eu_dist2_torch(
                            DATA[i:i+1],
                            DATA[j:j+1],
                            sqrt=False
                        )[0, 0].item()

        rho = np.exp(-DD / beta)
        W[i, j] = rho * np.exp(rho + 1)

    W = W.tocsr()

    # -------- Global weight matrix Wn --------
    # 注意：原代码中 rho 是最后一次循环的 rho（MATLAB 也是这样）
    rho_global = rho * np.exp(rho + 1)
    Wn = rho_global * np.ones((n, n))

    for i, j in zip(A_i, A_j):
        Wn[i, j] = 0

    # ======================================================
    # =============== Local Laplacian (Tensor) ==============
    # ======================================================
    D = np.array(W.sum(axis=1)).flatten()
    L_np = sp.diags(D) - W
    L_np = L_np.toarray()
    L = torch.tensor(L_np, dtype=torch.float32, device=device)

    # ======================================================
    # ============== Global Laplacian (Tensor) ==============
    # ======================================================
    Wn_t = torch.tensor(Wn, dtype=torch.float32, device=device)
    Dn = torch.sum(Wn_t, dim=1)
    Ln = torch.diag(Dn) - Wn_t

    # print('Computing Laplacian eigenvectors.')

    return L, Ln



def m_locaglob(DATA, TYPE='nn', PARAM=6):
    """
    Python version of locaglob.m
    Returns:
        L  : local Laplacian
        Ln : global Laplacian
    """
    DATA = np.asarray(DATA, dtype=np.float32)
    n = DATA.shape[0] #(300, 4000)

    print('\nLaplacian Eigenmaps Embedding.')

    # adjacency (sparse)
    A = adjacency(DATA, TYPE, PARAM)
    W = A.copy().tolil()

    # find non-zero entries  #所有相连的点
    A_i, A_j = A.nonzero()

    # beta 全局尺度，控制高斯核的带宽
    D_full = eu_dist2(DATA, sqrt=False)
    beta = np.mean(np.sum(D_full, axis=1))

    # -------- Local weight matrix W --------
    for i, j in zip(A_i, A_j):
        DD = eu_dist2(
                        DATA[i:i+1], 
                        DATA[j:j+1],   #i和j之间的欧式距离
                        sqrt=False)[0, 0]  
        
        rho = np.exp(-DD / beta) #权重公式
        W[i, j] = rho * np.exp(rho + 1)

    W = W.tocsr()

    # -------- Global weight matrix Wn --------
    # start with full matrix
    rho_global = rho * np.exp(rho + 1)
    Wn = rho_global * np.ones((n, n))

    for i, j in zip(A_i, A_j):
        Wn[i, j] = 0

    # -------- Local Laplacian --------
    D = np.array(W.sum(axis=1)).flatten()
    L = sp.diags(D) - W
    L = L.toarray()

    # -------- Global Laplacian --------
    Dn = np.sum(Wn, axis=1)
    Ln = np.diag(Dn) - Wn

    # print('Computing Laplacian eigenvectors.')

    return L, Ln


