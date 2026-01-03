from torch.autograd import Function
import torch
import torch.nn as nn

from MKGLMFA.manifoldfeature import manifea_troch

class ReverseLayerF(Function):
    # 这段代码通过继承 torch.autograd.Function 自定义了一个自动求导函数。

    # 正向传播
    @staticmethod  # 是 Python 的装饰器，用来声明一个静态方法（static method）。
    def forward(ctx, x, alpha):
        # 操作：它不对输入 x 做任何实质性修改，只是通过 x.view_as(x) 返回了与 x 相同的张量。
        
        # 参数：alpha 是一个超参数，用于控制梯度反转的强度。
        # 存储：ctx.alpha = alpha 将该参数保存在上下文中，供反向传播时使用。
        ctx.alpha = alpha

        return x.view_as(x)

    # 反向传播（核心所在）
    @staticmethod
    def backward(ctx, grad_output):
        # neg()：将梯度取反（乘以 -1）。
        # * ctx.alpha：对梯度进行缩放。
        output = grad_output.neg() * ctx.alpha  
        # 返回值：返回 output, None。因为 forward 有两个输入（x 和 alpha），所以反向传播也要返回两个梯度。
        # alpha 是超参数不需要梯度，所以返回 None。
        return output, None


class ManifoldLayer_BatchLevel(nn.Module):
    """
    batch level
    Non-parametric manifold projection layer (MKGLMFA).
    This layer is intentionally NON-differentiable.
    """

    def __init__(self, options):
        super().__init__()
        self.options = options

    def forward(self, feature, labels):
        """
        feature: Tensor [B, D], requires_grad=True
        labels : Tensor [B]
        return : Tensor [B, d]
        """

        device = feature.device

        # ---------- 强制不进计算图 ----------
        with torch.no_grad():
            feature_np = feature.detach().cpu()
            labels_np = labels.detach().cpu()

            Z = manifea_troch(feature_np, labels_np, self.options)

            if not torch.is_tensor(Z):
                Z = torch.from_numpy(Z)

            Z = Z.to(device)

        return Z
    
def manifold_reg_loss(feature, L):
    """
    feature: [B, d], requires_grad=True
    L: [B, B] Laplacian (no grad)
    """
    return torch.trace(feature.T @ L @ feature)