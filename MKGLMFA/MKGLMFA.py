
import numpy as np
import scipy.sparse as sp
import scipy.linalg as la
import time
import torch

from .distance_calculate import eu_dist2
from .distance_calculate import eu_dist2_torch
from .kernel import construct_kernel
from .kernel import construct_kernel_torch

def RRKGE_torch(W, D, options, data, Ln, L, device=None):
    if options is None:
        options = {}

    # Dim = options.get('ReducedDim', 30)
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

    # # ---------- labels (ALWAYS CPU) ----------
    # if torch.is_tensor(gnd):
    #     gnd = gnd.detach().cpu().numpy()
    # else:
    #     gnd = np.asarray(gnd)

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
    # rho = np.exp(-DD[:, 0] / beta)
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
        I = flat_idx // nSmp
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


# ------------------------no torch version--------------------------------

def MKGLMFA(gnd, data, Ln, L, options=None):
    """
    Python version of MKGLMFA.m
    """
    if options is None:
        options = {}

    nSmp = data.shape[0]
    labels = np.unique(gnd)
    nLabel = len(labels)

    # ---------- Kernel ---------- K 的每一行，表示“第 i 个样本与所有样本的相似度向量”
    K, timeK = construct_kernel(data, None, options)  #把原始特征映射到 Reproducing Kernel Hilbert Space（RKHS）

    # ---------- Parameters ----------
    intraK = options.get('intraK', 5)
    interK = options.get('interK', 20)

    # ---------- Distance ----------
    D = eu_dist2(K, sqrt=False)         # 两个样本在“核相似度表示空间”中的距离
    beta = np.mean(np.sum(D, axis=1))

    # ---------- Intra-class graph Sc ----------类内
    Sc = sp.lil_matrix((nSmp, nSmp))
    nIntraPair = 0

    for lab in labels:
        idx = np.where(gnd == lab)[0]
        D_class = D[np.ix_(idx, idx)]
        order = np.argsort(D_class, axis=1) # 在“类内”距离矩阵上，找每个样本的最近邻

        nClass = len(idx)
        nIntraPair += nClass ** 2

        if intraK < nClass:
            order = order[:, :intraK + 1]  # 截取 前intraK（10）
        else:
            last = order[:, -1][:, None]
            order = np.hstack([order, np.repeat(last, intraK + 1 - nClass, axis=1)])

        for i, row in enumerate(idx):
            for j in order[i]:
                col = idx[j]
                Sc[row, col] = 1

    I, J = Sc.nonzero()
    DD = eu_dist2(K[I], K[J], sqrt=False)
    rho = np.exp(-DD[:, 0] / beta)  ## 为什么取第一列？
    Sc[I, J] = rho * np.exp(rho + 1)
    Sc = Sc.maximum(Sc.T)  # Sc[i, j] = max(Sc[i, j], Sc[j, i])

    # ---------- Inter-class graph Sp ----------类间
    if interK > 0 and interK < (nSmp ** 2 - nIntraPair):
        D_tmp = D.copy()
        maxD = D.max() + 100

        for lab in labels:
            idx = np.where(gnd == lab)[0]
            D_tmp[np.ix_(idx, idx)] = maxD #把属于同一类的，赋值一个 大数字, 和其他类的还是保持和空间的距离

        flat_idx = np.argsort(D_tmp.ravel())[:interK] # 因为同类已经赋值大数字，排序排到最后
        I, J = np.unravel_index(flat_idx, (nSmp, nSmp))# 所有样本中 ，选出 interK 个 类间样本对 (i, j)，它们在核空间中距离最小

        DD = eu_dist2(K[I], K[J], sqrt=False)
        rho = np.exp(-DD[:, 0] / beta)

        Sp = sp.coo_matrix(
            (rho * np.exp(rho - 1), (I, J)), shape=(nSmp, nSmp)
        ).tocsr()
        Sp = Sp.maximum(Sp.T)

    else:
        Sp = np.ones((nSmp, nSmp))
        for lab in labels:
            idx = np.where(gnd == lab)[0]
            Sp[np.ix_(idx, idx)] = 0
        Sp = sp.csr_matrix(Sp)

    # ---------- Laplacians ----------
    # 类内拉普拉斯
    Dc = np.array(Sc.sum(axis=1)).flatten()
    Sc = -Sc
    Sc.setdiag(Sc.diagonal() + Dc)

    # 类间拉普拉斯
    Dp = np.array(Sp.sum(axis=1)).flatten()
    Sp = -Sp
    Sp.setdiag(Sp.diagonal() + Dp)

    # ---------- Kernel centering ----------
    if not options.get('keepMean', False):  # 核矩阵的“去均值 / 中心化（kernel centering）(把坐标原点放在数据中心，坐标才能旋转)”
        K = K - K.mean(axis=0, keepdims=True)  #等价于 做了一半的中心化（右乘 H）：
                                               # 为什么不是完整的 HKH？因为在 后面的 RRKGE 里，又做了一次完整的中心化：
    # ---------- RRKGE ----------
    eigvector, eigvalue, elapse = RRKGE(Sp, Sc, options, K, Ln, L)

    # ---------- Clean small eigenvalues ----------
    mask = eigvalue >= 1e-10
    eigvalue = eigvalue[mask]
    eigvector = eigvector[:, mask]

    return eigvector, eigvalue, elapse


def RRKGE(W, D, options, data, Ln, L):
    if options is None:
        options = {}

    Dim = options.get('ReducedDim', 30)     # 降维后的特征维度
    Regu = options.get('Regu', 0)               # 是否在矩阵中加正则项
    ReguAlpha = options.get('ReguAlpha', 0.01)      # 正则项的权重系数（trade-off

    # Kernel matrix
    if options.get('Kernel', 0):
        K = data.copy()
        K = np.maximum(K, K.T)
        timeK = 0
    else:
        K, timeK = construct_kernel(data, None, options)

    nSmp = K.shape[0]

    # -------- Centering kernel --------
    t0 = time.process_time()

    sumK = np.sum(K, axis=1, keepdims=True)
    Kc = K - sumK / nSmp - sumK.T / nSmp + np.sum(sumK) / (nSmp ** 2)  # 完整的去中心化
    Kc = np.maximum(Kc, Kc.T)

    timePCA = time.process_time() - t0

    # -------- PCA or Regularized form --------
    t0 = time.process_time()

    if not Regu:
        eigval_pca, eigvec_pca = la.eigh(Kc)
        idx = eigval_pca / np.max(np.abs(eigval_pca)) > 1e-6
        eigval_pca = eigval_pca[idx]
        eigvec_pca = eigvec_pca[:, idx]

        Kp = eigvec_pca

        Dp = Kp.T @ D @ Kp if D is not None else None

        Wp = Kp.T @ W @ Kp
        Wp = Wp + options.get('ReguBeta', 0) * (Kp.T @ Ln @ Kp)
        Dp = Dp + ReguAlpha * (Kp.T @ L @ Kp)

    else:
        Wp = Kc.T @ W @ Kc      # 这是把 类间结构（Sp / W） 投影到核子空间。
        Wp = Wp + options.get('ReguBeta', 0) * (Kc.T @ Ln @ Kc)     # 往“类间目标”里加入一个“全局流形分离项”
        Dp = Kc.T @ D @ Kc + ReguAlpha * (Kc.T @ L @ Kc)        # 把“局部流形保持”放进“约束项”里

    Wp = np.maximum(Wp, Wp.T)       # 强制把 Wp 和 Dp 变成“数值对称矩阵”
    Dp = np.maximum(Dp, Dp.T)

    # -------- Generalized eigen --------
    dimMatrix = Wp.shape[0]
    Dim = min(Dim, dimMatrix)

    eigval, eigvec = la.eigh(Wp, Dp)
    idx = np.argsort(-eigval)
    eigval = eigval[idx][:Dim]
    eigvec = eigvec[:, idx][:, :Dim]

    if not Regu:
        eigvec = Kp @ (eigvec / eigval_pca[:, None])

    # -------- Normalize --------
    norm = np.sqrt(np.sum((eigvec.T @ K) * eigvec.T, axis=1))
    eigvec = eigvec / norm

    timeMethod = time.process_time() - t0

    elapse = {
        'timeK': timeK,
        'timePCA': timePCA,
        'timeMethod': timeMethod,
        'timeAll': timeK + timePCA + timeMethod
    }

    return eigvec, eigval, elapse
