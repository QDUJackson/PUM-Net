import matplotlib.pyplot as plt
import os
import torch
import torchvision.utils as vutils



def save_test_samples(
        epoch: int,
        batch_data: dict,
        model1_pred: torch.Tensor,
        grid_test: torch.Tensor,
        global_indices: list,
        det_value: torch.Tensor,
        root_dir: str = "saved_samples"):

    SELECTED_TEST_INDICES = set(range(30))

    # ==== 1️⃣  创建本 epoch 的总目录，并准备统计文件句柄 ====
    epoch_dir = os.path.join(root_dir, f"epoch_{epoch:03d}")
    os.makedirs(epoch_dir, exist_ok=True)
    stats_file = os.path.join(epoch_dir, "det_stats.txt")

    # 如果想每次重写而不是追加，把 "a" 改为 "w"
    with open(stats_file, "a", encoding="utf-8") as sf:

        # ==== 2️⃣  逐样本处理 ====
        for i, idx in enumerate(global_indices):
            if idx not in SELECTED_TEST_INDICES:
                continue

            # 子目录：epoch_xxx/idx_yyyyy/
            sample_dir = os.path.join(epoch_dir, f"idx_{idx:05d}")
            os.makedirs(sample_dir, exist_ok=True)

            # —— 保存相关图片 ——
            vutils.save_image(batch_data["model1"][i].cpu(),
                              os.path.join(sample_dir, "t1_warp.png"))
            vutils.save_image(batch_data["model2"][i].cpu(),
                              os.path.join(sample_dir, "t2.png"))
            vutils.save_image(batch_data["model_normal"][i].cpu(),
                              os.path.join(sample_dir, "t1.png"))
            vutils.save_image(model1_pred[i].cpu(),
                              os.path.join(sample_dir, "t1_hat.png"))
            vutils.save_image(grid_test[i].cpu(),
                              os.path.join(sample_dir, "Grid.png"))

            # —— det_value 可视化 ——
            det_img = det_value[i].squeeze().detach().cpu().numpy()
            plt.figure(figsize=(4, 4))
            plt.imshow(det_img, cmap="bwr")
            plt.colorbar(label="det_value")
            plt.title(f"det (idx {idx})")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(sample_dir, "det_value.png"),
                        dpi=300, bbox_inches="tight")
            plt.close()

            # —— 统计最小/最大并写入汇总文件（每行一个 idx） ——
            det_min = float(det_img.min())
            det_max = float(det_img.max())
            sf.write(f"idx {idx:05d}\tmin {det_min:.6f}\tmax {det_max:.6f}\n")

# ==================== Checkpoint ==================== #
def save_checkpoint(
        save_path,
        epoch,
        model,
        optimizer,
        best_dice_loss,
        train_loss_list,
        test_loss_list,
        train_MI_Loss_list,
        test_MI_Loss_list,
        train_DiceLoss_list,
        test_DiceLoss_list,
        train_MSE_Loss_list,
        test_MSE_Loss_list,
        epochs_list,
        train_det_mean_list,
        test_det_mean_list,
        train_det_max_list,
        test_det_max_list,
        train_det_min_list,
        test_det_min_list
):
    """
    仅保存与 det / 损失 曲线相关的数据，已删除全部 theta 信息
    """
    state = {
        'epoch': epoch,
        'best_dice_loss': best_dice_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss_list': train_loss_list,
        'test_loss_list': test_loss_list,
        'train_MI_Loss_list': train_MI_Loss_list,
        'test_MI_Loss_list': test_MI_Loss_list,
        'train_DiceLoss_list': train_DiceLoss_list,
        'test_DiceLoss_list': test_DiceLoss_list,
        'train_MSE_Loss_list': train_MSE_Loss_list,
        'test_MSE_Loss_list': test_MSE_Loss_list,
        'epochs_list': epochs_list,
        'train_det_mean_list': train_det_mean_list,
        'test_det_mean_list': test_det_mean_list,
        'train_det_max_list': train_det_max_list,
        'test_det_max_list': test_det_max_list,
        'train_det_min_list': train_det_min_list,
        'test_det_min_list': test_det_min_list
    }

    checkpoint_path = os.path.join(save_path, 'latest.pth')
    torch.save(state, checkpoint_path)
    print(f"===>> Checkpoint saved: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer):
    """
    读取与 save_checkpoint 对应的简化字段
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    best_dice_loss = checkpoint['best_dice_loss']
    completed_epoch = checkpoint['epoch']

    train_loss_list = checkpoint.get('train_loss_list', [])
    test_loss_list = checkpoint.get('test_loss_list', [])
    train_MI_Loss_list = checkpoint.get('train_MI_Loss_list', [])
    test_MI_Loss_list = checkpoint.get('test_MI_Loss_list', [])
    train_DiceLoss_list = checkpoint.get('train_DiceLoss_list', [])
    test_DiceLoss_list = checkpoint.get('test_DiceLoss_list', [])
    train_MSE_Loss_list = checkpoint.get('train_MSE_Loss_list', [])
    test_MSE_Loss_list = checkpoint.get('test_MSE_Loss_list', [])
    epochs_list = checkpoint.get('epochs_list', [])

    train_det_mean_list = checkpoint.get('train_det_mean_list', [])
    test_det_mean_list = checkpoint.get('test_det_mean_list', [])
    train_det_max_list = checkpoint.get('train_det_max_list', [])
    test_det_max_list = checkpoint.get('test_det_max_list', [])
    train_det_min_list = checkpoint.get('train_det_min_list', [])
    test_det_min_list = checkpoint.get('test_det_min_list', [])

    print(f"===>> Checkpoint loaded: {checkpoint_path} (Last completed epoch: {completed_epoch})")

    return (best_dice_loss, completed_epoch,
            train_loss_list, test_loss_list,
            train_MI_Loss_list, test_MI_Loss_list,
            train_DiceLoss_list, test_DiceLoss_list,
            train_MSE_Loss_list, test_MSE_Loss_list,
            epochs_list,
            train_det_mean_list, test_det_mean_list,
            train_det_max_list, test_det_max_list,
            train_det_min_list, test_det_min_list)


def save_best_model(model, dice_loss, epoch, save_path, best_dice_loss):
    if dice_loss < best_dice_loss:
        best_dice_loss = dice_loss
        torch.save(model.state_dict(), os.path.join(save_path, 'best_dice_model_swin.pth'))
        print('*************************************************************************')
        print(f'New best model saved: Epoch {epoch}, DiceLoss: {dice_loss:.5f}')
        print('*************************************************************************')
    return best_dice_loss


# ==================== Curves ==================== #
import os
import matplotlib.pyplot as plt

def update_plots(
        epoch,
        save_path,
        epochs_list,
        train_DiceLoss_list, test_DiceLoss_list,
        train_MSE_Loss_list,  test_MSE_Loss_list,
        train_loss_list,      test_loss_list,
        train_DET_Loss_list,  test_DET_Loss_list
):
    """
    4 张曲线：Dice / MSE / Total / DET_Loss
    - 自动处理 x/y 长度不一致
    - 自动把标量 / numpy / torch.Tensor 转成 list
    """

    def _to_list(a):
        # 把各种可能的类型转成 Python list[float]
        if a is None:
            return []
        if isinstance(a, list):
            return a
        if isinstance(a, tuple):
            return list(a)
        try:
            import numpy as np
            if isinstance(a, np.ndarray):
                return a.reshape(-1).tolist()
        except Exception:
            pass
        try:
            import torch
            if isinstance(a, torch.Tensor):
                return a.detach().flatten().cpu().tolist()
        except Exception:
            pass
        # 标量
        try:
            return [float(a)]
        except Exception:
            return []

    def _safe_plot(ax, x, y, label):
        x = _to_list(x)
        y = _to_list(y)
        if len(y) == 0:
            return
        # 用 y 的长度作为基准；如果 x 太短，就生成 1..len(y) 的横轴
        if len(x) < len(y):
            x = list(range(1, len(y) + 1))
        else:
            x = x[:len(y)]
        ax.plot(x, y, label=label)

    plt.figure(figsize=(16, 12))

    # 1. Dice
    ax = plt.subplot(2, 2, 1)
    _safe_plot(ax, epochs_list, train_DiceLoss_list, 'Train Dice')
    _safe_plot(ax, epochs_list, test_DiceLoss_list,  'Test Dice')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Dice Coefficient'); ax.set_title('Dice Coefficient Curve')
    ax.legend(); ax.grid(True)

    # 2. MSE
    ax = plt.subplot(2, 2, 2)
    _safe_plot(ax, epochs_list, train_MSE_Loss_list, 'Train MSE Loss')
    _safe_plot(ax, epochs_list, test_MSE_Loss_list,  'Test MSE Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE Loss'); ax.set_title('MSE Loss Curve')
    ax.legend(); ax.grid(True)

    # 3. Total Loss
    ax = plt.subplot(2, 2, 3)
    _safe_plot(ax, epochs_list, train_loss_list, 'Train Total Loss')
    _safe_plot(ax, epochs_list, test_loss_list,  'Test Total Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Total Loss'); ax.set_title('Total Loss Curve')
    ax.legend(); ax.grid(True)

    # 4. DET Loss
    ax = plt.subplot(2, 2, 4)
    _safe_plot(ax, epochs_list, train_DET_Loss_list, 'Train DET Loss')
    _safe_plot(ax, epochs_list, test_DET_Loss_list,  'Test DET Loss')
    ax.set_xlabel('Epoch'); ax.set_ylabel('DET Loss'); ax.set_title('DET Loss Curve')
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, 'latest.png'))
    plt.close()


# ==================== Utils ==================== #
def update_learning_rate(optimizer, epoch):
    if epoch % 6 == 0 and epoch < 16:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.7
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch}: Learning rate is {current_lr}")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
