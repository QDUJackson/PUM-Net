# -*- coding: utf-8 -*-
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.transforms as T

from MIDataSet import MultimodalRegistrationDataset
from module.MID_GaussVar import JacksonNet, SpatialTransformer
from module.utils_train import (
    save_checkpoint, load_checkpoint, save_best_model,
    update_plots,         # 仅作占位，实际未再调用
    update_learning_rate, count_parameters,    # pylint: disable=unused-import
    save_test_samples
)
from module.utils import ( label_to_one_hot)
from losses import (
    NCCLoss, dice_coefficient,
    laplace_loss
)
from module.utils import label_value_to_index

import time
# ===================== 熵正则：工具函数（新增） =====================
def alpha_neg_entropy_reg(model, temperature: float = 1.0):
    """
    遍历模型中具有 alpha_logits 的 MultiSigmaGaussianSmoother，
    计算它们的“负熵”之和： sum(p * log p)，并返回 (reg, 模块数)。
    说明：
      - 使用 log_softmax，数值稳定；
      - temperature>1 可让分布更平滑（等价于对 logits 除以 T）。
    """
    nmods = 1
    z = model.smooth.alpha_logits / float(temperature)

    log_p = F.log_softmax(z,dim=-1)
    p = log_p.exp()
    reg = (p * log_p).sum()  # 负熵（<=0）

    return reg, nmods


@torch.no_grad()
def print_smoother_alphas(model, prefix=""):
    """
    打印模型里每个 MultiSigmaGaussianSmoother 的 alpha（softmax 后）。
    便于监控是否塌缩到某一两个尺度。
    """
    lines = []
    for name, m in model.named_modules():
        if getattr(m, "alpha_logits", None) is not None:
            p = torch.softmax(m.alpha_logits, dim=-1)
            lines.append(f"{prefix}{name}: " + " ".join([f"{v.item():.6f}" for v in p]))
        elif hasattr(m, "alpha_fixed"):
            p = m.alpha_fixed
            lines.append(f"{prefix}{name} (fixed): " + " ".join([f"{v.item():.6f}" for v in p]))
    if lines:
        print("\n".join(lines))


# ============================== 训练入口 ==============================
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])

    save_path = 'CAW_318_LD'
    os.makedirs(save_path, exist_ok=True)
    print(f"Path is {save_path}")

    train_data_path = os.path.join(save_path, 'train_data.txt')
    test_data_path = os.path.join(save_path, 'test_data.txt')

    # ---------------------------------------------------------
    # 数据集路径
    # ---------------------------------------------------------
    base_dir = 'data8k20/'
    train_model1_dir        = os.path.join(base_dir, 'train/t1_warp')
    train_model1_normal_dir = os.path.join(base_dir, 'train/t1')
    train_model2_dir        = os.path.join(base_dir, 'train/t2')
    train_label_dir         = os.path.join(base_dir, 'train/seg')
    train_seg_dir           = os.path.join(base_dir, 'train/seg_warp')

    test_model1_dir         = os.path.join(base_dir, 'val/t1_warp')
    test_model1_normal_dir  = os.path.join(base_dir, 'val/t1')
    test_model2_dir         = os.path.join(base_dir, 'val/t2')
    test_label_dir          = os.path.join(base_dir, 'val/seg')
    test_seg_dir            = os.path.join(base_dir, 'val/seg_warp')

    dataset_train = MultimodalRegistrationDataset(
        train_model1_dir, train_model2_dir, train_label_dir,
        train_seg_dir, train_model1_normal_dir, transform=transform
    )
    dataset_test = MultimodalRegistrationDataset(
        test_model1_dir, test_model2_dir, test_label_dir,
        test_seg_dir, test_model1_normal_dir, transform=transform
    )

    batch_size   = 8
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True,  drop_last=True)
    test_loader  = DataLoader(dataset_test,  batch_size=batch_size, shuffle=False, drop_last=True)

    device = "cuda" if torch.cuda.is_available() else 'cpu'

    model_net = JacksonNet().to(device)
    print(f"Model total number of parameters: {count_parameters(model_net):,}")

    optimizer    = torch.optim.Adam(model_net.parameters(), lr=8e-4)
    ncc_loss_fn  = NCCLoss()
    miu1         = 200
    num_epochs   = 120

    # --------- 熵正则超参（新增） ---------
    alpha_entropy_weight = 0   # γ：负熵系数（1e-4 ~ 1e-2 之间调）
    alpha_temperature    = 1.0    # T：温度（>1 分布更平）

    # ========== 统计列表 ==========
    train_DiceLoss_list, test_DiceLoss_list = [], []
    train_MSE_Loss_list,  test_MSE_Loss_list  = [], []
    train_loss_list,      test_loss_list      = [], []
    train_DET_Loss_list,  test_DET_Loss_list  = [], []
    epochs_list                                   = []

    # （以下 det_* 列表保留，仅用于 checkpoint，程序中不再更新/使用）
    train_det_mean_list, test_det_mean_list = [], []
    train_det_max_list,  test_det_max_list  = [], []
    train_det_min_list,  test_det_min_list  = [], []

    best_dice_loss, start_epoch = float('inf'), 0

    # ========= 断点续训 =========
    checkpoint_resume_path = os.path.join(save_path, 'latest.pth')
    if os.path.exists(checkpoint_resume_path):
        (best_dice_loss, completed_epoch,
         train_loss_list, test_loss_list,
         _, _,   # MI loss 列表占位
         train_DiceLoss_list, test_DiceLoss_list,
         train_MSE_Loss_list, test_MSE_Loss_list,
         epochs_list,
         train_det_mean_list, test_det_mean_list,
         train_det_max_list,  test_det_max_list,
         train_det_min_list,  test_det_min_list) = load_checkpoint(
            checkpoint_resume_path, model_net, optimizer)
        start_epoch = completed_epoch

    # -------- 灰度网格图 --------
    transform_gray = T.Compose([
        T.Resize((256, 256)),
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ])
    single_img_path = 'data8k10/Grid/Grid.png'
    if not os.path.isfile(single_img_path):
        raise FileNotFoundError(f"{single_img_path} 不存在，请检查路径")

    Grid_tensor = (
        transform_gray(Image.open(single_img_path))
        .to(device)
        .unsqueeze(0)   # [1,1,256,256]
        .repeat(batch_size, 1, 1, 1)
    )
    STN = SpatialTransformer((256, 256)).to(device)
    det_value = torch.zeros((8,1,256,256), device=device)

    # =================== 训练循环 ===================
    for epoch in range(start_epoch, num_epochs):
        current_epoch = epoch + 1
        print(f"\n===== Epoch {current_epoch} / {num_epochs} =====")
        update_learning_rate(optimizer, current_epoch)

        # ---------------- Training ----------------
        model_net.train()
        epoch_train_loss = epoch_train_diceloss = epoch_train_MSE_loss = epoch_train_DET_loss = 0.0

        for data in train_loader:
            model1_img        = data['model1'].to(device)
            model2_img        = data['model2'].to(device)
            label             = data['label'].to(device) * 255.0
            seg               = data['seg'].to(device)
            model1_normal_img = data['model_normal'].to(device)

            optimizer.zero_grad()

            torch.cuda.synchronize() if device == "cuda" else None
            start_time = time.time()

            flow_phi = model_net(model1_img, model2_img)

            torch.cuda.synchronize() if device == "cuda" else None
            end_time = time.time()
            print(f"Forward time is {end_time - start_time:.3f}s")

            model1_hat = STN(model1_img, flow_phi)

            gradient_loss_value = laplace_loss(flow_phi, penalty='l2')
            MSE_Loss            = ncc_loss_fn(model1_hat, model1_normal_img)

            total_loss          = gradient_loss_value + miu1 * MSE_Loss

            '''# -------- 熵正则（新增） --------
            neg_entropy_reg, nmods = alpha_neg_entropy_reg(model_net, temperature=alpha_temperature)
            if nmods > 0:
                total_loss = total_loss + alpha_entropy_weight * neg_entropy_reg  # 注：neg_entropy<=0'''

            # ------- segmentation 指标统计 -------
            seg_source_idx = label_value_to_index(seg * 255.0)
            seg_source_1h  = label_to_one_hot(seg_source_idx, num_classes=4)
            seg_wrapped    = STN(seg_source_1h, flow_phi)
            seg_idx        = torch.argmax(seg_wrapped, dim=1)  # [B,H,W]
            mapping        = torch.tensor([0, 64, 128, 255.0], device=seg_idx.device)
            seg_discrete   = mapping[seg_idx].unsqueeze(1)

            epoch_train_diceloss += dice_coefficient(seg_discrete, label).item()
            epoch_train_MSE_loss += MSE_Loss.item()
            epoch_train_loss     += total_loss.item()

            total_loss.backward()
            optimizer.step()

        # 每个 epoch 打印一次当前 α（便于监控是否塌缩）
        print_smoother_alphas(model_net, prefix=f"[epoch {current_epoch}] ")

        # ---------------- Testing ----------------
        model_net.eval()
        epoch_test_loss = epoch_test_diceloss = epoch_test_MSE_loss = epoch_test_DET_loss = 0.0

        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                model1_img        = data['model1'].to(device)
                model2_img        = data['model2'].to(device)
                label             = data['label'].to(device) * 255.0
                seg               = data['seg'].to(device)
                model1_normal_img = data['model_normal'].to(device)

                flow_phi   = model_net(model1_img, model2_img)
                model1_hat = STN(model1_img, flow_phi)

                gradient_loss_value = laplace_loss(flow_phi, penalty='l2')
                MSE_Loss            = ncc_loss_fn(model1_hat, model1_normal_img)

                total_loss          = gradient_loss_value + miu1 * MSE_Loss

                '''# —— 测试阶段是否加入熵正则：可选（此处加入，保证 train/test loss 定义一致）
                neg_entropy_reg, nmods = alpha_neg_entropy_reg(model_net, temperature=alpha_temperature)
                if nmods > 0:
                    total_loss = total_loss + alpha_entropy_weight * neg_entropy_reg'''

                # ------- segmentation 指标统计 -------
                seg_source_idx = label_value_to_index(seg * 255.0)
                seg_source_1h  = label_to_one_hot(seg_source_idx, num_classes=4)
                seg_wrapped    = STN(seg_source_1h, flow_phi)
                seg_idx        = torch.argmax(seg_wrapped, dim=1)  # [B,H,W]
                mapping        = torch.tensor([0, 64.0, 128.0, 255.0], device=seg_idx.device)
                seg_discrete   = mapping[seg_idx].unsqueeze(1)

                epoch_test_diceloss += dice_coefficient(seg_discrete, label).item()
                epoch_test_MSE_loss += MSE_Loss.item()
                epoch_test_loss     += total_loss.item()

                # ---------- 保存可视化 ----------
                start_idx  = batch_idx * batch_size
                grid_test  = STN(Grid_tensor, flow_phi)
                save_test_samples(
                    current_epoch, data, model1_hat, grid_test,
                    list(range(start_idx, start_idx + model1_img.size(0))),
                    det_value,
                    root_dir=os.path.join(save_path, "selected_test_samples")
                )

        # -------- 统计 --------
        num_train_batches = len(train_loader)
        num_test_batches  = len(test_loader)

        train_stats = {
            'loss': epoch_train_loss / num_train_batches,
            'dice': epoch_train_diceloss / num_train_batches,
            'mse':  epoch_train_MSE_loss / num_train_batches,
            'det':  epoch_train_DET_loss / num_train_batches
        }
        test_stats = {
            'loss': epoch_test_loss / num_test_batches,
            'dice': epoch_test_diceloss / num_test_batches,
            'mse':  epoch_test_MSE_loss / num_test_batches,
            'det':  epoch_test_DET_loss / num_test_batches
        }

        # -------- 记录列表 --------
        epochs_list.append(current_epoch)

        train_DiceLoss_list.append(train_stats['dice'])
        test_DiceLoss_list.append(test_stats['dice'])

        train_MSE_Loss_list.append(train_stats['mse'])
        test_MSE_Loss_list.append(test_stats['mse'])

        train_DET_Loss_list.append(train_stats['det'])
        test_DET_Loss_list.append(test_stats['det'])

        train_loss_list.append(train_stats['loss'])
        test_loss_list.append(test_stats['loss'])

        # -------- 打印 --------
        print(f"Training -> Loss: {train_stats['loss']:.5f}, Dice: {train_stats['dice']:.5f}, "
              f"MSE: {train_stats['mse']:.5f}, DET: {train_stats['det']:.5f}")
        print(f"Test     -> Loss: {test_stats['loss']:.5f}, Dice: {test_stats['dice']:.5f}, "
              f"MSE: {test_stats['mse']:.5f}, DET: {test_stats['det']:.5f}")

        # -------- 写文件 --------
        with open(train_data_path, 'a') as f:
            f.write(f"{current_epoch}\t{train_stats['dice']:.5f}\t{train_stats['loss']:.5f}\t"
                    f"{train_stats['det']:.5f}\n")
        with open(test_data_path, 'a') as f:
            f.write(f"{current_epoch}\t{test_stats['dice']:.5f}\t{test_stats['loss']:.5f}\t"
                    f"{test_stats['det']:.5f}\n")

        # -------- 绘图 --------
        update_plots(
            current_epoch, save_path, epochs_list,
            train_DiceLoss_list, test_DiceLoss_list,
            train_MSE_Loss_list,  test_MSE_Loss_list,
            train_loss_list,      test_loss_list,
            train_DET_Loss_list,  test_DET_Loss_list
        )

        best_dice_loss = save_best_model(
            model_net, test_stats['dice'], current_epoch, save_path, best_dice_loss
        )

        # -------- 保存 checkpoint --------
        save_checkpoint(
            save_path, current_epoch, model_net, optimizer, best_dice_loss,
            train_loss_list, test_loss_list,
            [], [],   # MI loss 列表占位
            train_DiceLoss_list, test_DiceLoss_list,
            train_MSE_Loss_list, test_MSE_Loss_list,
            epochs_list,
            train_det_mean_list, test_det_mean_list,
            train_det_max_list,  test_det_max_list,
            train_det_min_list,  test_det_min_list
        )

    print("Training complete!")
