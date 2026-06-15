import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy
import torch.nn.functional as F
def diff_periodic_central(x: torch.Tensor, spacing: float = 1.0):

    if x.dim() != 4:
        raise ValueError("x 必须是形状 (N,C,H,W) 的 4-D 张量")

    inv_2h = 0.5 / spacing

    # ---------- 竖直方向中心差分 ----------
    # 上移一格（y+1）：最后一行拼到顶部
    x_forward = torch.cat([x[:, :, 1:, :], x[:, :, :1, :]], dim=2)
    # 下移一格（y-1）：第一行拼到末尾
    x_backward = torch.cat([x[:, :, -1:, :], x[:, :, :-1, :]], dim=2)
    x1 = (x_forward - x_backward) * inv_2h  # ∂/∂y

    # ---------- 水平方向中心差分 ----------
    # 右移一格（x+1）：最右列拼到最左
    x_forward = torch.cat([x[:, :, :, 1:], x[:, :, :, :1]], dim=3)
    # 左移一格（x-1）：最左列拼到最右
    x_backward = torch.cat([x[:, :, :, -1:], x[:, :, :, :-1]], dim=3)
    x2 = (x_forward - x_backward) * inv_2h  # ∂/∂x

    return x1, x2
def label_value_to_index_leaf(label):
    """
    将标签从 {0,64,128,255} 映射到 {0,1,2,3}.
    label: [B,H,W] 或 [B,1,H,W]
    返回:  [B,H,W], 值为 {0,1,2,3}
    """
    if label.dim() == 4 and label.size(1) == 1:
        label = label.squeeze(1)
    out = torch.zeros_like(label)
    out[label == 20] = 1
    out[label == 39] = 2
    out[label == 59] = 3
    out[label == 78] = 4
    out[label == 98] = 5
    out[label == 118] = 6
    out[label == 137] = 7
    out[label == 157] = 8
    out[label == 177] = 9
    out[label == 196] = 10
    out[label == 216] = 11
    out[label == 235] = 12
    out[label == 255] = 13
    return out


def label_value_to_index(label):
    """
    将标签从 {0,64,128,255} 映射到 {0,1,2,3}.
    label: [B,H,W] 或 [B,1,H,W]
    返回:  [B,H,W], 值为 {0,1,2,3}
    """
    if label.dim() == 4 and label.size(1) == 1:
        label = label.squeeze(1)
    out = torch.zeros_like(label)
    out[label == 64] = 1
    out[label == 128] = 2
    out[label == 255] = 3
    return out


def label_to_one_hot(label_idx, num_classes=4):
    """
    将 [B,H,W] 的类别索引转换为 One-hot 格式 [B,4,H,W].
    """
    one_hot = F.one_hot(label_idx.long(), num_classes=num_classes)  # [B,H,W,4]
    one_hot = one_hot.permute(0, 3, 1, 2).float()  # [B,4,H,W]
    return one_hot
def binary_label_value_to_index(label):
    """
    将标签从 {0,255} 映射到 {0,1}
    label: [B,H,W] 或 [B,1,H,W]
    返回:  [B,H,W], 值为 {0,1}
    """
    if label.dim() == 4 and label.size(1) == 1:
        label = label.squeeze(1)

    out = torch.zeros_like(label, dtype=torch.long)
    out[label == 255] = 1
    return out

def gumbel_soft_threshold(x, value_set, temperature=1.0):
    x_expanded = x.unsqueeze(-1)  # (N, C, H, W, 1)
    value_set = torch.tensor(value_set).to(x.device).float()  # (K,)

    # 计算负的绝对距离
    distances = -torch.abs(x_expanded - value_set)  # (N, C, H, W, K)

    # 添加 Gumbel 噪声
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(distances) + 1e-20) + 1e-20)
    logits = (distances + gumbel_noise) / temperature

    # 计算 Gumbel-Softmax 权重
    weights = F.softmax(logits, dim=-1)  # (N, C, H, W, K)

    # 计算加权和
    soft_quantized = (weights * value_set).sum(dim=-1)  # (N, C, H, W)

    return soft_quantized


def check_values(seg_wrapped_discrete):
    """
    检查 seg_wrapped_discrete 中的所有唯一值是否属于 [0, 64, 128, 255]。

    参数：
        seg_wrapped_discrete (Tensor): 待检查的分割图像张量。

    返回：
        int: 如果所有唯一值属于 [0, 64, 128, 255]，返回 1；否则返回 0。
    """
    unique_values = torch.unique(seg_wrapped_discrete)
    expected_values = torch.tensor([0, 64, 128, 255], device=seg_wrapped_discrete.device)

    # 检查 unique_values 是否全部在 expected_values 中
    # torch.isin 返回一个与 expected_values 大小相同的布尔张量
    isin = torch.isin(unique_values, expected_values)

    # 如果所有 unique_values 都在 expected_values 中，flag = 1，否则 flag = 0
    flag = torch.all(isin).int().item()

    return flag
def total_check_values(seg_wrapped_discrete):

    for counts in range(100):
        flag1 = check_values(seg_wrapped_discrete)

        if flag1 == 0:
            seg_wrapped_discrete = gumbel_soft_threshold(seg_wrapped_discrete, [0, 64, 128, 255], 0.0000001)
        else:
            break
        if counts > 90:
            print(f'Counts is {counts}, oh my!!!')

    return seg_wrapped_discrete

def total_check_values_No_print(seg_wrapped_discrete):

    for counts in range(100):
        flag1 = check_values(seg_wrapped_discrete)
        if flag1 == 0:
            seg_wrapped_discrete = gumbel_soft_threshold(seg_wrapped_discrete, [0, 64, 128, 255], 0.0000001)
        else:
            break

    return seg_wrapped_discrete

def soft_threshold(x, value_set, temperature=0.00000001):
    # 将输入扩展以匹配 value_set 的形状
    x_expanded = x.unsqueeze(-1)  # 形状变为 (N, C, H, W, 1)
    value_set = value_set.clone().detach().requires_grad_(True).to(x.device).float()
    # 计算距离
    distances = -torch.abs(x_expanded - value_set)  # 形状为 (N, C, H, W, K)
    # 计算 softmax 权重
    weights = F.softmax(distances / temperature, dim=-1)  # 同上
    # 计算加权和
    soft_quantized = (weights * value_set).sum(dim=-1)  # 结果形状为 (N, C, H, W)
    return soft_quantized

def warp_images_grid_sample(I2w, u, v):

    batch_size, _, height, width = I2w.shape
    # Create a normalized grid
    #print(f"height {height} width {width}")
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, height - 1, height, device=I2w.device),
        torch.linspace(0, width - 1, width, device=I2w.device),
        indexing='ij'
    )

    # Repeat the grid for the batch size
    grid_y = grid_y.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, height, width]
    grid_x = grid_x.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, height, width]
    #print(f"grid y shape {grid_y.shape} u shape {u.shape}")
    # Apply the displacements to the grid
    y_shift = grid_y + u.squeeze(1)  # Shape: [batch_size, height, width]
    x_shift = grid_x + v.squeeze(1)  # Shape: [batch_size, height, width]

    y_shift = torch.clamp(y_shift, 0, height - 1)
    x_shift = torch.clamp(x_shift, 0, width - 1)

    # Normalize the grid to the range [-1, 1]
    y_shift = 2 * (y_shift / (height - 1)) - 1  # Shape: [batch_size, height, width]
    x_shift = 2 * (x_shift / (width - 1)) - 1  # Shape: [batch_size, height, width]

    # Combine grids into a single grid for grid_sample
    grid = torch.stack((x_shift, y_shift), dim=-1)  # Shape: [batch_size, height, width, 2]

    # Use grid_sample to warp the images
    warped_images = F.grid_sample(I2w, grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped_images

def extract_dynamic_patches_with_grid_sample(t1_images, t2_images, seg_images, margin=16):
    """
    Extract patches using grid_sample with dynamically adjusted sizes based on tumor region.
    Args:
        t1_images (torch.Tensor): T1 modality images, shape [B, C, H, W].
        t2_images (torch.Tensor): T2 modality images, shape [B, C, H, W].
        seg_images (torch.Tensor): Segmentation masks, shape [B, C, H, W].
        margin (int): Margin around the tumor region.
    Returns:
        torch.Tensor, torch.Tensor: Padded T1 and T2 modality patches, shape [B, C, max_patch_height, max_patch_width].
    """
    B, C, H, W = t1_images.shape
    t1_patches = []
    t2_patches = []

    max_patch_height = 0
    max_patch_width = 0

    for i in range(B):
        seg = seg_images[i, 0]  # Use channel dimension 0 for segmentation mask
        non_zero_coords = torch.nonzero(seg > 0, as_tuple=False)
        # 这个地方是非零元素的位置

        if non_zero_coords.size(0) > 0:
            # Get tumor region boundaries
            y_min = torch.min(non_zero_coords[:, 0]).item()
            y_max = torch.max(non_zero_coords[:, 0]).item()
            x_min = torch.min(non_zero_coords[:, 1]).item()
            x_max = torch.max(non_zero_coords[:, 1]).item()

            # Expand with margin
            x_min = max(0, x_min - margin)
            y_min = max(0, y_min - margin)
            x_max = min(W, x_max + margin)
            y_max = min(H, y_max + margin)
            # margin就是给肿瘤区域加边界

            # Define the grid for sampling
            patch_height = y_max - y_min
            patch_width = x_max - x_min

            max_patch_height = max(max_patch_height, patch_height)
            max_patch_width = max(max_patch_width, patch_width)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(y_min / (H - 1) * 2 - 1, y_max / (H - 1) * 2 - 1, patch_height, device=t1_images.device),
                torch.linspace(x_min / (W - 1) * 2 - 1, x_max / (W - 1) * 2 - 1, patch_width, device=t1_images.device),
                indexing="ij"
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)  # Shape [1, patch_height, patch_width, 2]

            # Use grid_sample to extract patches
            t1_patch = F.grid_sample(t1_images[i:i + 1], grid, mode='bilinear', align_corners=True)
            t2_patch = F.grid_sample(t2_images[i:i + 1], grid, mode='bilinear', align_corners=True)
            t1_patches.append(t1_patch)
            t2_patches.append(t2_patch)
        else:
            # No tumor region, return empty tensors
            t1_patches.append(torch.zeros((1, C, 1, 1), device=t1_images.device))
            t2_patches.append(torch.zeros((1, C, 1, 1), device=t2_images.device))

    # Pad all patches to the maximum size
    padded_t1_patches = []
    padded_t2_patches = []

    for t1_patch, t2_patch in zip(t1_patches, t2_patches):
        t1_padded = F.pad(t1_patch, (0, max_patch_width - t1_patch.shape[3], 0, max_patch_height - t1_patch.shape[2]))
        t2_padded = F.pad(t2_patch, (0, max_patch_width - t2_patch.shape[3], 0, max_patch_height - t2_patch.shape[2]))
        padded_t1_patches.append(t1_padded)
        padded_t2_patches.append(t2_padded)

    t1_patches_tensor = torch.cat(padded_t1_patches, dim=0)
    t2_patches_tensor = torch.cat(padded_t2_patches, dim=0)

    return t1_patches_tensor, t2_patches_tensor

def vision_patches(t1_patches,t2_patches):

    for i, (t1_patch, t2_patch) in enumerate(zip(t1_patches, t2_patches)):
        t1_patch = t1_patch.squeeze(0).cpu().numpy()  # Remove channel dimension
        t2_patch = t2_patch.squeeze(0).cpu().numpy()  # Remove channel dimension

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(t1_patch, cmap='gray')
        plt.title(f"T1 Tumor Patch {i}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(t2_patch, cmap='gray')
        plt.title(f"T2 Tumor Patch {i}")
        plt.axis('off')

        plt.show()



