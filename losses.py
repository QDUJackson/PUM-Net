"""
*Preliminary* pytorch implementation.

Losses for VoxelMorph
"""

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_multi_class(pred_softmax, target_onehot, smooth=1e-5):
    """
    多分类 Dice Loss (可微)
    pred_softmax: [B,4,H,W], 已经过 softmax 的预测概率
    target_onehot: [B,4,H,W], one-hot 标签
    """
    intersection = (pred_softmax * target_onehot).sum(dim=(2, 3))  # [B,4]
    denominator = (pred_softmax.pow(2) + target_onehot.pow(2)).sum(dim=(2, 3))  # [B,4]
    dice = (2. * intersection + smooth) / (denominator + smooth)  # [B,4]
    dice_mean_batch = dice.mean(dim=0)  # [4]
    loss = 1.0 - dice_mean_batch.mean()  # 标量，loss 越小越好
    return loss
def det_loss(x, eps=1e-5, scale = 1e1):
    return scale * F.relu(eps - x).mean()
def laplace_loss(x, penalty='l2'):
    # 只取内部像素，边缘像素自动舍弃
    center = x[:, :, 1:-1, 1:-1]
    up     = x[:, :, :-2, 1:-1]
    down   = x[:, :,  2:, 1:-1]
    left   = x[:, :, 1:-1, :-2]
    right  = x[:, :, 1:-1,  2:]

    lap = -4.0 * center + up + down + left + right   # 离散 Laplace

    # L1 / L2 惩罚
    if penalty == 'l1':
        lap = torch.abs(lap)
    else:  # 'l2'
        lap = lap * lap

    eps = 1e-8
    return torch.mean(lap) + eps
# def gradient_loss(s, penalty='l2'):
#     dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
#     dx = torch.abs(s[:, :, :, 1:, :] - s[:, :, :, :-1, :])
#     dz = torch.abs(s[:, :, :, :, 1:] - s[:, :, :, :, :-1])

#     if (penalty == 'l2'):
#         dy = dy * dy
#         dx = dx * dx
#         dz = dz * dz

#     d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
#     return d / 3.0

def MI_loss(H):

    h = H.sum(dim=2, keepdim=True)  # 对 width 维度求和，结果形状为 (batch_size, 1, 256, 1)
    h1 = H.sum(dim=3, keepdim=True)  # 对 height 维度求和，结果形状为 (batch_size, 1, 1, 256)
    # 进一步调整形状
    h = h.squeeze(dim=2)  # 结果形状为 (batch_size, 1, 256)
    h1 = h1.squeeze(dim=3)  # 结果形状为 (batch_size, 1, 256)
    # 其中h,h1都是边缘概率密度
    nonzero_indices = H > 1e-15
    # 这里找的是联合概率密度中不为0的点

    # 使用广播机制计算每个批次的互信息
    H_nonzero = H[nonzero_indices]

    h1_expanded = h1.unsqueeze(3)  # 形状 (batch_size, 1, 256, 1)
    h_expanded = h.unsqueeze(2)  # 形状 (batch_size, 1, 1, 256)

    denom = (h1_expanded * h_expanded)[nonzero_indices]
    '''
    注意：
    强度联合概率密度不为0，是不可能存在对应的边缘概率密度为0的
    '''
    s1_nonzero = H_nonzero * torch.log(H_nonzero / (denom))

    # 将互信息映射回原始形状的张量中
    s1 = torch.zeros_like(H)
    s1[nonzero_indices] = s1_nonzero

    # 对每个批次的结果进行求和
    s1 = s1.sum(dim=(1, 2, 3))
    s1_mean = s1.mean()
    return -1.0 * s1_mean

def gradient_loss(s, penalty='l2'):
    # dy = torch.abs(s[:, :, 1:, :, :] - s[:, :, :-1, :, :])
    dx = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
    dz = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

    epss = 1e-8

    if (penalty == 'l2'):
        # dy = dy * dy
        dx = dx * dx
        dz = dz * dz

    d = torch.mean(dx) + torch.mean(dz) + epss
    return d / 2.0
def laplace_squared_loss(s, eps=1e-8):
    """
    计算输入张量的Laplace算子的平方（即二阶导数的平方和）作为正则化损失
    
    参数:
        s: 输入张量，形状为 (batch_size, channels, height, width)
        eps: 小常数用于数值稳定性
        
    返回:
        laplace_sq: Laplace算子的平方和损失
    """
    # 计算二阶导数 (中心差分)
    # x方向二阶导: f(x+1) + f(x-1) - 2f(x)
    d2x = s[:, :, 2:, :] + s[:, :, :-2, :] - 2 * s[:, :, 1:-1, :]
    
    # y方向二阶导: f(y+1) + f(y-1) - 2f(y)
    d2y = s[:, :, :, 2:] + s[:, :, :, :-2] - 2 * s[:, :, :, 1:-1]
    
    # 裁剪使得尺寸匹配（因为二阶导计算后尺寸会减小）
    s_cropped = s[:, :, 1:-1, 1:-1]
    
    # Laplace算子 = d2x + d2y
    laplace = d2x[:, :, :, 1:-1] + d2y[:, :, 1:-1, :]
    
    # Laplace平方
    laplace_sq = torch.mean(laplace**2) + eps
    
    return laplace_sq
def mse_loss(x, y):
    return torch.mean((x - y) ** 2)

def dice_coefficient(pred, target, epsilon=1e-6):
    # 只考虑大于0的点
    pred_nonzero = pred > 0
    target_nonzero = target > 0

    # 计算交集：两个图像在每个像素上的值相等且大于0
    intersection = ((pred == target) & (pred_nonzero & target_nonzero)).float().sum(dim=(1, 2, 3))

    # 计算A和B：pred和target中大于0的点数
    A = pred_nonzero.float().sum(dim=(1, 2, 3))
    B = target_nonzero.float().sum(dim=(1, 2, 3))

    # 计算Dice系数
    dice = (2 * intersection + epsilon) / (A + B + epsilon)

    # 返回负的平均Dice系数作为损失
    return -dice.mean()

class NCCLoss(nn.Module):
    def __init__(self, eps=1e-5, reduction='mean'):
        super(NCCLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, img1, img2):
        img1 = img1.float()
        img2 = img2.float()

        B = img1.size(0)

        mean1 = torch.mean(img1, dim=(1, 2, 3), keepdim=True)
        mean2 = torch.mean(img2, dim=(1, 2, 3), keepdim=True)

        std1 = torch.std(img1, dim=(1, 2, 3), unbiased=False, keepdim=True)
        std2 = torch.std(img2, dim=(1, 2, 3), unbiased=False, keepdim=True)

        img1_norm = (img1 - mean1) / (std1 + self.eps)
        img2_norm = (img2 - mean2) / (std2 + self.eps)

        ncc = torch.sum(img1_norm * img2_norm, dim=(1, 2, 3)) / (img1.size(1) * img1.size(2) * img1.size(3))

        loss = 1.0-ncc

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def cross_modal_contrastive_loss(feat_t1, feat_t2, normalize=True, temperature=1.0):
    """
    用于多模态对比学习的Loss：
    L = -1/B * Σ_{m=1..B} log( exp(s(x_m, y_m)) / [ Σ exp(s(x_m, y_n)) + Σ exp(s(x_m, x_n)) ] )

    参数：
        feat_t1: [B, 1, H, W] 或者可被 view 为 [B, d] 的张量 (模态X，如T1)
        feat_t2: [B, 1, H, W] 或者可被 view 为 [B, d] 的张量 (模态Y，如T2)
        normalize: (bool) 是否对 flatten 后的特征做 L2 归一化 (常见对比学习做法)
        temperature: (float) 温度系数 (相似度会除以该值)；若过小会导致数值爆炸；通常0.07~0.2之间

    返回：
        标量 loss (float Tensor)，可反向传播
    """

    B = feat_t1.shape[0]  # batch_size

    # 1) 拉平 [B, 1, H, W] -> [B, d]
    f1 = feat_t1.view(B, -1)  # (B, d)
    f2 = feat_t2.view(B, -1)  # (B, d)

    # 2) 可选：L2归一化
    if normalize:
        f1 = F.normalize(f1, p=2, dim=1)  # (B, d)
        f2 = F.normalize(f2, p=2, dim=1)  # (B, d)

    # 3) 计算相似度矩阵
    # sim_xy[m, n] = 点积( f1[m], f2[n] ) / temperature
    # sim_xx[m, n] = 点积( f1[m], f1[n] ) / temperature
    sim_xy = torch.matmul(f1, f2.t()) / temperature  # [B,B]
    sim_xx = torch.matmul(f1, f1.t()) / temperature  # [B,B]

    # 4) exponentiate
    exp_xy = torch.exp(sim_xy)  # [B,B]
    exp_xx = torch.exp(sim_xx)  # [B,B]

    # 分子 = 正对相似度（对角）
    # pos[m] = exp_xy[m,m]
    pos = torch.diagonal(exp_xy)  # (B,)

    # 5) 分母 = sum_{n=1..B}(exp_xy[m,n]) + sum_{n=1..B}(exp_xx[m,n])
    sum_xy = torch.sum(exp_xy, dim=1)  # [B]
    sum_xx = torch.sum(exp_xx, dim=1)  # [B]
    denominator = sum_xy + sum_xx      # [B]

    # 6) 为防止 denominator == 0 或 pos == 0，避免 log(0)-> -inf
    eps = 1e-8
    ratio = pos / (denominator + eps)   # 避免分母为0
    ratio = torch.clamp(ratio, min=eps) # 避免 ratio=0

    # 7) loss_i = -log( ratio )
    # overall loss = mean(loss_i)
    loss_each = -torch.log(ratio)
    loss = torch.mean(loss_each)

    return loss


import torch
import torch.nn.functional as F

def multimodal_contrastive_loss_4d(zx, zy, temperature=1.0, eps=1e-8):
    """
    基于 -MSE 的多模态对比学习损失函数，适用于任意 >= 2D 的输入；
    实际做法是先 flatten => (B, D)，再做两两比对。

    参数：
    ----------
    zx : torch.Tensor, shape = [B, ...]
        表示批大小 B 个样本在模态 x (T1) 下的特征/图像。
        后面所有维度都将被展平到 D。
    zy : torch.Tensor, shape = [B, ...]
        表示批大小 B 个样本在模态 y (T2) 下的特征/图像。
    temperature : float
        温度系数 t，用于缩放相似度。
    eps : float
        防止数值过小导致 log(0) 的溢出。

    返回：
    ----------
    loss : torch.Tensor (标量)
        该批数据上计算得到的总损失 L = Lx + Ly。
    """
    B = zx.size(0)

    # 1) 先将输入展平 => [B, D]
    zx_flat = zx.view(B, -1)  # [B, D]
    zy_flat = zy.view(B, -1)  # [B, D]

    # 2) 计算 pairwise MSE => 距离矩阵 dist_xx, dist_xy, dist_yy => [B,B]
    #    dist_xx[m,n] = MSE(zx_flat[m], zx_flat[n])
    #    其中 zx_flat[m] => shape (D,)
    #    为了方便：采用 (u - v)^2 再对 dim=-1 平均

    dist_xx = (zx_flat[:, None] - zx_flat[None, :]) ** 2  # [B,1,D] - [1,B,D] => [B,B,D]
    dist_xx = dist_xx.mean(dim=-1)                       # => [B,B]

    dist_xy = (zx_flat[:, None] - zy_flat[None, :]) ** 2
    dist_xy = dist_xy.mean(dim=-1)

    dist_yy = (zy_flat[:, None] - zy_flat[None, :]) ** 2
    dist_yy = dist_yy.mean(dim=-1)

    # 3) 将距离转换为相似度: s(u,v) = -MSE(u,v)
    sim_xx = -dist_xx  # [B,B]
    sim_xy = -dist_xy  # [B,B]
    sim_yy = -dist_yy  # [B,B]

    # 4) 对相似度做 exp(sim / temperature)
    exp_xx = torch.exp(sim_xx / temperature)
    exp_xy = torch.exp(sim_xy / temperature)
    exp_yy = torch.exp(sim_yy / temperature)

    # 5) 分别计算 Lx 和 Ly
    #    Lx_m = -log( exp_xy[m,m] / ( sum_{n != m} (exp_xy[m,n] + exp_xx[m,n]) ) )
    Lx_list = []
    for m in range(B):
        numerator_m = exp_xy[m, m]
        denominator_m = (exp_xy[m, :] + exp_xx[m, :]).sum() \
                        - (exp_xy[m, m] + exp_xx[m, m])
        Lx_m = -torch.log((numerator_m + eps) / (denominator_m + eps))
        Lx_list.append(Lx_m)
    Lx = torch.stack(Lx_list).mean()

    #    Ly_m = -log( exp_yx[m,m] / ( sum_{n != m} (exp_yx[m,n] + exp_yy[m,n]) ) )
    #    其中 exp_yx = exp_xy^T
    exp_yx = exp_xy.transpose(0, 1)
    Ly_list = []
    for m in range(B):
        numerator_m = exp_yx[m, m]
        denominator_m = (exp_yx[m, :] + exp_yy[m, :]).sum() \
                        - (exp_yx[m, m] + exp_yy[m, m])
        Ly_m = -torch.log((numerator_m + eps) / (denominator_m + eps))
        Ly_list.append(Ly_m)
    Ly = torch.stack(Ly_list).mean()

    loss = Lx + Ly
    return loss

def compute_gradient(img):
    """
    计算图像在 x/y 方向上的梯度近似（简单差分）。
    输入 img: [B, 1, H, W]
    返回 (dx, dy)，与 img 同形状
    """
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]

    # 补齐维度，右边/下边补 0
    dx = F.pad(dx, (0, 1, 0, 0))  # 在W方向右侧补1列
    dy = F.pad(dy, (0, 0, 0, 1))  # 在H方向下侧补1行

    return dx, dy

def l2_loss(a, b):
    """计算 a, b 的 L2 范数差异 (MSE)"""
    return F.mse_loss(a, b)

def l2_grad_loss(t1,t2,z1,z2):

    dx_t1,dy_t1 = compute_gradient(t1)
    dx_t2,dy_t2 = compute_gradient(t2)
    dx_z1,dy_z1 = compute_gradient(z1)
    dx_z2,dy_z2 = compute_gradient(z2)


    loss0 = l2_loss(dx_t2,dx_z1) + l2_loss(dy_t2,dy_z1)
    loss = l2_loss(dx_t1, dx_z2) + l2_loss(dy_t1, dy_z2)

    return loss + loss0

def activation_decay(embs, p=2.0, eps=1e-8):
    """
    对若干输出张量(embedding)做 L^p 范数衰减 (按像素/元素平均)。
    embs: list[Tensor] or tuple[Tensor]
    p: 幂次 (1.0 表示L1, 2.0 表示L2)
    """
    total_sum = 0.0
    total_elems = 0

    for x in embs:
        val = x.abs().pow(p)
        total_sum += val.sum()
        total_elems += val.numel()

    return (total_sum + eps) / (total_elems + eps)

def total_decay_loss(z1,z2):
    decay_l2 = activation_decay([z1, z2], p=2.0)
    decay_l1 = activation_decay([z1, z2], p=1.0)
    decay_loss = decay_l2 + decay_l1
    return decay_loss


# ------------------ 测试用例 (简单测试) ------------------
'''if __name__ == "__main__":
    # 假设我们有一个 batch size = 4, channels=1, height=16, width=16
    # 但也有可能是 [4, 1, 1, 16, 16]，都不影响，我们会直接 flatten。
    B, C, H, W = 4, 1, 16, 16
    zx_test = torch.randn(B, C, H, W)      # shape [4, 1, 16, 16]
    zy_test = torch.randn(B, C, H, W)      # shape [4, 1, 16, 16]

    loss_val = multimodal_contrastive_loss_4d(zx_test, zy_test, temperature=0.5)
    print("multimodal_contrastive_loss_4d =", loss_val.item())'''

