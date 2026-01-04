
import numpy as np
import scipy.sparse as sp
import torch

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

    # DATA = DATA.double()
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

def adjacency(DATA, TYPE='nn', PARAM=6):
    """
    Python version of adjacency.m
    Returns a symmetric sparse adjacency matrix A (NxN)
    """
    DATA = np.asarray(DATA, dtype=np.float32)
    n = DATA.shape[0]

    A = sp.lil_matrix((n, n))
    step = 100

    for i1 in range(0, n, step):
        i2 = min(i1 + step, n)
        XX = DATA[i1:i2]  # 100,4000

        # Squared L2 distance
        aa = np.sum(XX * XX, axis=1, keepdims=True)
        bb = np.sum(DATA * DATA, axis=1, keepdims=True).T
        dt = aa + bb - 2 * (XX @ DATA.T)
        dt = np.maximum(dt, 0) #分块中每一个样本到数据集所有样本的欧氏距离的平方

        # sort distances 排序
        idx = np.argsort(dt, axis=1)

        if TYPE == 'nn':
            for i in range(i1, i2): #当前分块的全局索引（在整个数据集的位置）
                row = i - i1 # 全局行号转成分块行号
                for j in range(1, PARAM + 1): #跳过自己j=0,去取距离最近的 PARAM个邻居
                    jj = idx[row, j] # 第row行，按照升序排列的，全局 列号 列表，jj就是第j个最近邻居的样本编号
                    val = np.sqrt(dt[row, jj])
                    A[i, jj] = val  #边i to jj：权重设置为val
                    A[jj, i] = val  #反向边jj to i：权重也设置为val，矩阵对称

        elif TYPE in ('eps', 'epsballs'):
            for i in range(i1, i2):
                row = i - i1
                j = 1
                while j < n and np.sqrt(dt[row, idx[row, j]]) < PARAM:
                    jj = idx[row, j]
                    val = np.sqrt(dt[row, jj])
                    A[i, jj] = val
                    A[jj, i] = val
                    j += 1
        else:
            raise ValueError("TYPE must be 'nn' or 'epsballs'")

    return A.tocsr() #

