# from .graph import m_locaglob
from .graph import m_locaglob_torch
# from .MKGLMFA import MKGLMFA
from .MKGLMFA import MKGLMFA_torch
# from .kernel import construct_kernel
from .kernel import construct_kernel_torch
import numpy as np
import torch

def manifea_troch(in_feature, in_label, options):
    """
    Non-differentiable manifold feature extraction (MKGLMFA)

    Parameters
    ----------
    in_feature : torch.Tensor [B, D]
    in_label   : torch.Tensor [B]
    options    : dict

    Returns
    -------
    manifold_feature : torch.Tensor [B, d]
    """

    device = in_feature.device

    # =========================================================
    # ❶ 明确：流形学习不参与反向传播
    # =========================================================
    with torch.no_grad():

        # -------- Local / Global Laplacian --------
        L, Ln = m_locaglob_torch(
            in_feature,
            TYPE='nn',
            PARAM=5,
            device=device
        )

        # -------- MKGLMFA --------
        eigvector, eigvalue, _ = MKGLMFA_torch(
            gnd=in_label,
            data=in_feature,
            Ln=Ln,
            L=L,
            options=options,
            device=device
        )

        # 数值安全
        eigvector = torch.nan_to_num(eigvector, nan=0.0)

        # -------- Kernel (torch version!) --------
        Ktrain, _ = construct_kernel_torch(
            in_feature,
            None,
            options
        )

        Ktrain = torch.nan_to_num(Ktrain, nan=0.0)

        # -------- Projection --------
        manifold_feature = Ktrain @ eigvector   # [B, ReducedDim]
        manifold_feature = manifold_feature.float()

    return manifold_feature

