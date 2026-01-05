import numpy as np
from .distance_calculate import eu_dist2
from .distance_calculate import eu_dist2_torch

import time
import torch

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
    fea_a = fea_a.double()

    if device is not None:
        fea_a = fea_a.to(device)

    if fea_b is not None:
        if not torch.is_tensor(fea_b):
            fea_b = torch.tensor(fea_b, dtype=torch.float64)
        fea_b = fea_b.double()
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















def construct_kernel(fea_a, fea_b=None, options=None):
    """
    Python version of constructKernel.m
    """
    if options is None:
        options = {}

    kernel_type = options.get('KernelType', 'Gaussian').lower()

    t0 = time.process_time()

    if kernel_type == 'gaussian':
        t = options.get('t', 1)
        if fea_b is None:
            D = eu_dist2(fea_a, sqrt=False)
        else:
            D = eu_dist2(fea_a, fea_b, sqrt=False)
        K = np.exp(-D / (2 * t * t))

    elif kernel_type == 'polynomial':
        d = options.get('d', 2)
        K = fea_a @ fea_a.T if fea_b is None else fea_a @ fea_b.T
        K = K ** d

    elif kernel_type == 'polyplus':
        d = options.get('d', 2)
        K = fea_a @ fea_a.T if fea_b is None else fea_a @ fea_b.T
        K = (K + 1) ** d

    elif kernel_type == 'linear':
        K = fea_a @ fea_a.T if fea_b is None else fea_a @ fea_b.T

    else:
        raise ValueError("Unknown KernelType")

    if fea_b is None:
        K = np.maximum(K, K.T)

    elapse = time.process_time() - t0
    return K, elapse
