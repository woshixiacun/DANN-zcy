import numpy as np
import scipy.sparse as sp
import torch
import time

def adjacency_torch(DATA, TYPE='nn', PARAM=6, step=100, device=None):
    """
    PyTorch version of adjacency.m
    Returns a symmetric sparse adjacency matrix A (NxN)

    Args:
        DATA: (N, d) numpy array or torch tensor
        TYPE: 'nn' or 'epsballs'
        PARAM: k for knn ("nn"), epsilon for epsballs
        step: block size
        device: 'cpu' or 'cuda'
    """
    if not torch.is_tensor(DATA):
        DATA = torch.tensor(DATA, dtype=torch.float32)

    if device is not None:
        DATA = DATA.to(device)

    n = DATA.shape[0]

    A = sp.lil_matrix((n, n))

    # 预先算好所有样本的 ||x||^2
    bb = torch.sum(DATA * DATA, dim=1, keepdim=True).t()  # (1, n)

    for i1 in range(0, n, step):
        i2 = min(i1 + step, n)
        XX = DATA[i1:i2]  # (step, d) #分块计算

        # squared Euclidean distance
        aa = torch.sum(XX * XX, dim=1, keepdim=True)  # (step, 1)
        dt = aa + bb - 2 * (XX @ DATA.t()) # dt[i][j]=|| xi-xj||^2
        dt = torch.clamp(dt, min=0) #防止浮点误差导致负值

        # sort distances
        idx = torch.argsort(dt, dim=1) 

        if TYPE == 'nn':
            for i in range(i1, i2): #k紧邻，对于每个样本，取最近的PARAM个近邻
                row = i - i1
                for j in range(1, PARAM + 1):  # skip itself
                    jj = idx[row, j].item()
                    val = torch.sqrt(dt[row, jj]).item()
                    A[i, jj] = val
                    A[jj, i] = val  #把权重设为欧式距离

        elif TYPE in ('eps', 'epsballs'):
            for i in range(i1, i2):
                row = i - i1
                j = 1
                while j < n:
                    jj = idx[row, j].item()
                    dist = torch.sqrt(dt[row, jj]).item()
                    if dist >= PARAM:
                        break
                    A[i, jj] = dist
                    A[jj, i] = dist
                    j += 1
        else:
            raise ValueError("TYPE must be 'nn' or 'epsballs'")

    return A.tocsr()  #返回稀疏矩阵，节省内存


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
        aa = torch.sum(fea_a * fea_a, dim=1, keepdim=True)  # (n, 1)
        bb = torch.sum(fea_b * fea_b, dim=1, keepdim=True)  # (m, 1)
        ab = fea_a @ fea_b.t()                              # (n, m)

        D = aa + bb.t() - 2 * ab
        D = torch.clamp(D, min=0)

        if sqrt:
            D = torch.sqrt(D)

        return D.abs()


def construct_kernel_torch(fea_a, fea_b=None, options=None, device=None):
    """
    PyTorch version of constructKernel.m

    Returns:
        K       : torch.Tensor
        elapse  : float (CPU time)
    """
    if options is None:
        options = {}

    kernel_type = options.get('KernelType', 'Gaussian').lower()

    t0 = time.process_time()

    # -------- tensor & device --------
    if not torch.is_tensor(fea_a):
        fea_a = torch.tensor(fea_a, dtype=torch.float64)

    if device is not None:
        fea_a = fea_a.to(device)

    if fea_b is not None:
        if not torch.is_tensor(fea_b):
            fea_b = torch.tensor(fea_b, dtype=torch.float64)

        if device is not None:
            fea_b = fea_b.to(device)

    # =====================================================
    # ================= Kernel types ======================
    # =====================================================
    if kernel_type == 'gaussian':
        t = options.get('t', 1)

        if fea_b is None:
            D = eu_dist2_torch(fea_a, sqrt=False)
        else:
            D = eu_dist2_torch(fea_a, fea_b, sqrt=False)

        K = torch.exp(-D / (2 * t * t))

    elif kernel_type == 'polynomial':
        d = options.get('d', 2)
        K = fea_a @ fea_a.t() if fea_b is None else fea_a @ fea_b.t()
        K = K ** d

    elif kernel_type == 'polyplus':
        d = options.get('d', 2)
        K = fea_a @ fea_a.t() if fea_b is None else fea_a @ fea_b.t()
        K = (K + 1) ** d

    elif kernel_type == 'linear':
        K = fea_a @ fea_a.t() if fea_b is None else fea_a @ fea_b.t()

    else:
        raise ValueError("Unknown KernelType")

    # -------- symmetry (fea_b is None) --------
    if fea_b is None:
        K = torch.maximum(K, K.t())

    elapse = time.process_time() - t0

    return K, elapse


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

    n = DATA.shape[0]  #8

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


def RRKGE_torch(W, D, options, data, Ln, L, device=None):
    if options is None:
        options = {}

    Regu = options.get('Regu', 0)
    ReguAlpha = options.get('ReguAlpha', 0.01)
    ReguBeta = options.get('ReguBeta', 0)

    # ---------- Kernel ----------
    if options.get('Kernel', 0):
        K = data.clone()
        K = torch.maximum(K, K.T)  #？？？？为什么做这一步，还原K，之前做了半中心化
    else:
        K, _ = construct_kernel_torch(data, None, options, device=device)

    nSmp = K.shape[0]

    # ---------- Full kernel centering ----------  全部去中心化，之前做了半取中心化
    sumK = torch.sum(K, dim=1, keepdim=True)
    Kc = K - sumK / nSmp - sumK.T / nSmp + torch.sum(sumK) / (nSmp ** 2)
    Kc = torch.maximum(Kc, Kc.T)
    Kc = Kc.to(W.dtype)
    Kc = Kc.to(device)

    # ---------- PCA / Regularized ----------
    if not Regu:
        eigval_pca, eigvec_pca = torch.linalg.eigh(Kc)
        mask = eigval_pca / eigval_pca.abs().max() > 1e-6
        eigval_pca = eigval_pca[mask]
        eigvec_pca = eigvec_pca[:, mask]

        Kp = eigvec_pca

        Wp = Kp.T @ W @ Kp
        Wp = Wp + ReguBeta * (Kp.T @ Ln @ Kp)

        Dp = Kp.T @ D @ Kp
        Dp = Dp + ReguAlpha * (Kp.T @ L @ Kp)

    else:
        Wp = Kc.T @ W @ Kc
        Wp = Wp + ReguBeta * (Kc.T @ Ln @ Kc)
        Dp = Kc.T @ D @ Kc + ReguAlpha * (Kc.T @ L @ Kc)

    # ---------- Symmetrize ----------
    Wp_global = torch.maximum(Wp, Wp.T)
    Dp_local = torch.maximum(Dp, Dp.T)

    return Wp_global, Dp_local
  

def MKGLMFA_torch(gnd, data, Ln, L, options=None, device=None):
    """
    PyTorch version of MKGLMFA.m

    Returns:
        eigvector : torch.Tensor (n x d)
        eigvalue  : torch.Tensor (d,)
        elapse    : dict
    """
    if options is None:
        options = {}

    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float32)

    if not torch.is_tensor(gnd):
        gnd = torch.tensor(gnd, dtype=torch.long)

    if device is not None:
        data = data.to(device)
        gnd = gnd.to(device)
        Ln = Ln.to(device)
        L = L.to(device)

    nSmp = data.shape[0]
    labels = torch.unique(gnd)

    # ---------- Kernel ----------
    K, _ = construct_kernel_torch(data, None, options, device=device)

    # ---------- Parameters ----------
    intraK = options.get('intraK', 5)
    interK = options.get('interK', 20)

    # ---------- Distance in RKHS ----------
    D = eu_dist2_torch(K, sqrt=False)
    beta = torch.mean(torch.sum(D, dim=1))

    # ---------- Intra-class graph Sc ----------
    Sc = torch.zeros((nSmp, nSmp), device=device)
    nIntraPair = 0

    for lab in labels:
        idx = torch.where(gnd == lab)[0]
        nClass = idx.numel()
        nIntraPair += nClass ** 2

        D_class = D[idx][:, idx]
        order = torch.argsort(D_class, dim=1)

        k = min(intraK + 1, nClass)
        order = order[:, :k]

        for i in range(nClass):
            row = idx[i]
            cols = idx[order[i]]
            Sc[row, cols] = 1.0

    I, J = torch.nonzero(Sc, as_tuple=True)
    DD = eu_dist2_torch(K[I], K[J], sqrt=False).squeeze()
    rho = torch.exp(-DD[:, 0] / beta)  # 什么取第一列？
    rho = rho.to(Sc.dtype)
    rho = rho.to(device)
    Sc[I, J] = rho * torch.exp(rho + 1)
    Sc = torch.maximum(Sc, Sc.T)

    # ---------- Inter-class graph Sp ----------
    if interK > 0 and interK < (nSmp ** 2 - nIntraPair):
        D_tmp = D.clone()
        maxD = D.max() + 100

        for lab in labels:
            idx = torch.where(gnd == lab)[0]
            D_tmp[idx[:, None], idx[None, :]] = maxD

        flat_idx = torch.argsort(D_tmp.flatten())[:interK]
        # I = flat_idx // nSmp
        I = torch.div(flat_idx, nSmp, rounding_mode='floor')
        J = flat_idx % nSmp

        DD = eu_dist2_torch(K[I], K[J], sqrt=False).squeeze()
        rho = torch.exp(-DD[:, 0]/ beta)   # 这里的rho是上一次迭代的最后一个值，而且是固定的？
        rho = rho.to(Sc.dtype)
        rho = rho.to(device)

        Sp = torch.zeros((nSmp, nSmp), device=device)
        Sp[I, J] = rho * torch.exp(rho - 1)
        Sp = torch.maximum(Sp, Sp.T)

    else:
        Sp = torch.ones((nSmp, nSmp), device=device)
        for lab in labels:
            idx = torch.where(gnd == lab)[0]
            Sp[idx[:, None], idx[None, :]] = 0.0

    # ---------- Laplacians ----------
    #计算Sc的拉普拉斯矩阵  类内
    Dc = torch.sum(Sc, dim=1)
    Sc = -Sc
    Sc.diagonal().add_(Dc)
    
    #计算Sp的拉普拉斯矩阵  类间
    Dp = torch.sum(Sp, dim=1)
    Sp = -Sp
    Sp.diagonal().add_(Dp)
    
    Sc_local = Sc
    Sp_global = Sp

    # ---------- Kernel centering ----------
    if not options.get('keepMean', False):
        K = K - K.mean(dim=0, keepdim=True)
    # ---------- RRKGE ----------
    Wp_global, Dp_local = RRKGE_torch(Sp, Sc, options, K, Ln, L, device=device)

    return Sc_local,Sp_global, Wp_global, Dp_local