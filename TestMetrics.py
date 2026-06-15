# -*- coding: utf-8 -*-
import os
import math
import csv
import torch
import numpy as np
import torch.nn as nn
from scipy.ndimage import distance_transform_edt as edt

from torch.utils.data import DataLoader
from torchvision import transforms

# === 你的项目内模块 ===
from MIDataSet import MultimodalRegistrationDataset
from module.MID_GaussVar import JacksonNet, SpatialTransformer
from losses import (NCCLoss, dice_coefficient)
from module.utils import (label_to_one_hot, label_value_to_index)
from module.utils_train import count_parameters


# ------------------ HD95 计算类 ------------------
class HausdorffDistance:
    '''CLASSES = [20.0, 39.0, 59.0, 78.0, 98.0,
               118.0, 137.0, 157.0, 177.0,
               196.0, 216.0, 235.0, 255.0]'''
    CLASSES = [64.0, 128.0, 255.0]

    @staticmethod
    def _hd95_distance(binary_x: np.ndarray, binary_y: np.ndarray) -> float:
        if binary_x.sum() == 0 and binary_y.sum() == 0:
            return 0.0
        if binary_x.sum() == 0 or binary_y.sum() == 0:
            return np.inf
        dist_map = edt(~binary_y.astype(bool))
        dists = dist_map[binary_x.astype(bool)]
        return 0.0 if dists.size == 0 else np.percentile(dists, 95)

    def compute(self, seg: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        assert seg.shape == label.shape and seg.dim() == 4, "seg/label 应为 [B,1,H,W]"
        seg_np = seg.cpu().numpy()
        label_np = label.cpu().numpy()
        B = seg_np.shape[0]
        results = []
        for b in range(B):
            hd_vals = []
            for v in self.CLASSES:
                pred_bin = (seg_np[b, 0] == v)
                gt_bin = (label_np[b, 0] == v)
                hd1 = self._hd95_distance(pred_bin, gt_bin)
                hd2 = self._hd95_distance(gt_bin, pred_bin)
                hd_vals.append(max(hd1, hd2))
            finite_vals = [d for d in hd_vals if not np.isinf(d)]
            results.append(max(finite_vals) if finite_vals else np.inf)
        return torch.tensor(results, dtype=torch.float, device=seg.device)


# ------------------ Gate 直方图统计器 ------------------
import matplotlib
matplotlib.use("Agg")  # 兼容无显示
import matplotlib.pyplot as plt

class GateHistograms:
    def __init__(self, bins=50, out_dir="gate_vis"):
        self.bins = int(bins)
        self.bin_edges = np.linspace(0.0, 1.0, self.bins + 1, dtype=np.float64)  # [0,1]
        self.counts = {}     # name -> np.ndarray(bins,)
        self.total = {}      # name -> int
        self.sumval = {}     # name -> float
        self.sumval2 = {}    # name -> float
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)

    def _ensure(self, name):
        if name not in self.counts:
            self.counts[name] = np.zeros(self.bins, dtype=np.float64)
            self.total[name] = 0
            self.sumval[name] = 0.0
            self.sumval2[name] = 0.0

    @torch.no_grad()
    def add_tensor(self, name, g: torch.Tensor):
        # g: [B,N,1] 或 [B,N]，值域 [0,1]
        self._ensure(name)
        if g.dim() == 3 and g.size(-1) == 1:
            g = g.squeeze(-1)
        elif g.dim() != 2:
            return
        arr = g.detach().to(dtype=torch.float32).cpu().numpy().reshape(-1)
        np.clip(arr, 0.0, 1.0, out=arr)
        hist, _ = np.histogram(arr, bins=self.bin_edges)
        self.counts[name] += hist
        self.total[name] += arr.size
        self.sumval[name] += float(arr.sum())
        self.sumval2[name] += float((arr.astype(np.float64) ** 2).sum())

    def _approx_percentile_from_hist(self, name, q):
        if name not in self.counts or self.total[name] == 0:
            return float("nan")
        target = q / 100.0 * self.total[name]
        cum = np.cumsum(self.counts[name])
        idx = np.searchsorted(cum, target, side="left")
        if idx <= 0:
            return float(self.bin_edges[0])
        if idx >= self.bins:
            return float(self.bin_edges[-1])
        prev_cum = cum[idx - 1]
        bin_count = self.counts[name][idx]
        if bin_count <= 0:
            return float(self.bin_edges[idx])
        frac = (target - prev_cum) / bin_count
        low = self.bin_edges[idx]
        high = self.bin_edges[idx + 1]
        return float(low + (high - low) * frac)

    def summarize_one(self, name):
        c = int(self.total.get(name, 0))
        if c == 0:
            return {
                "count": 0, "mean": float("nan"), "std": float("nan"),
                "p1": float("nan"), "p5": float("nan"),
                "p50": float("nan"), "p95": float("nan"), "p99": float("nan")
            }
        mean = self.sumval[name] / c
        var = max(self.sumval2[name] / c - mean * mean, 0.0)
        std = math.sqrt(var)
        return {
            "count": c, "mean": mean, "std": std,
            "p1": self._approx_percentile_from_hist(name, 1),
            "p5": self._approx_percentile_from_hist(name, 5),
            "p50": self._approx_percentile_from_hist(name, 50),
            "p95": self._approx_percentile_from_hist(name, 95),
            "p99": self._approx_percentile_from_hist(name, 99)
        }

    def save_layer_hist_png(self, name):
        if name not in self.counts:
            return
        counts = self.counts[name]
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2.0
        width = (self.bin_edges[1] - self.bin_edges[0]) * 0.9
        plt.figure(figsize=(6, 4))
        plt.bar(centers, counts, width=width)
        plt.title(f"Gate Histogram: {name}")
        plt.xlabel("g")
        plt.ylabel("count")
        plt.tight_layout()
        safe = name.replace("/", "_").replace(".", "_")
        png_path = os.path.join(self.out_dir, f"{safe}.png")
        plt.savefig(png_path, dpi=150)
        plt.close()

    def save_layer_hist_csv(self, name):
        if name not in self.counts:
            return
        safe = name.replace("/", "_").replace(".", "_")
        csv_path = os.path.join(self.out_dir, f"{safe}.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["bin_left", "bin_right", "count"])
            for i in range(self.bins):
                w.writerow([self.bin_edges[i], self.bin_edges[i + 1], int(self.counts[name][i])])

    def save_summary_csv(self):
        path = os.path.join(self.out_dir, "summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "count", "mean", "std", "p1", "p5", "p50", "p95", "p99"])
            for name in sorted(self.counts.keys()):
                s = self.summarize_one(name)
                w.writerow([
                    name, s["count"],
                    f"{s['mean']:.6f}", f"{s['std']:.6f}",
                    f"{s['p1']:.6f}", f"{s['p5']:.6f}",
                    f"{s['p50']:.6f}", f"{s['p95']:.6f}", f"{s['p99']:.6f}"
                ])

    def save_all(self):
        for name in self.counts.keys():
            self.save_layer_hist_png(name)
            self.save_layer_hist_csv(name)
        self.save_summary_csv()


# ------------------ 打印 smoother 学到的 alphas ------------------
def _print_smooth_alphas(model):
    # 优先 model.smooth；若无则全模型搜
    smooth_modules = []
    if hasattr(model, "smooth"):
        smooth_modules.append(("smooth", model.smooth))
    else:
        for name, m in model.named_modules():
            if hasattr(m, "sigmas") and hasattr(m, "_alphas"):
                smooth_modules.append((name, m))

    if not smooth_modules:
        print("⚠️ 未找到 MultiSigmaGaussianSmoother（没有 .smooth 或类似模块）。")
        return

    print("\n===== MultiSigmaGaussianSmoother learned alphas =====")
    for name, sm in smooth_modules:
        device = next(model.parameters()).device
        dtype = torch.float32
        with torch.no_grad():
            if getattr(sm, "alpha_logits", None) is not None:
                alphas_t = torch.softmax(sm.alpha_logits.to(device=device, dtype=dtype), dim=-1)
            else:
                alphas_t = sm.alpha_fixed.to(device=device, dtype=dtype)
        alphas = alphas_t.detach().cpu().numpy().tolist()
        sigmas = [float(s) for s in getattr(sm, "sigmas", [])]
        print(f"模块: {name}")
        print("sigmas :", sigmas)
        print("alphas :", [f"{a:.6f}" for a in alphas])
        for s, a in zip(sigmas, alphas):
            print(f"  sigma={s:>4.2f}  alpha={a:.6f}")
    print("================================\n")


# ------------------ 主程序 ------------------
if __name__ == "__main__":
    # ================ 0. 配置部分 ================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 8

    # 路径
    val_data_base = "data8k20/val"
    best_model_path = "CAW_318_LD/best_dice_model_swin.pth"

    # ================ 1. 数据准备 ================
    transform = transforms.Compose([transforms.ToTensor()])

    model1_dir = os.path.join(val_data_base, "t1_warp")
    model1_normal_dir = os.path.join(val_data_base, "t1")
    model2_dir = os.path.join(val_data_base, "t2")
    label_dir = os.path.join(val_data_base, "seg")
    seg_dir = os.path.join(val_data_base, "seg_warp")

    dataset_val = MultimodalRegistrationDataset(
        model1_dir, model2_dir, label_dir, seg_dir, model1_normal_dir, transform=transform
    )
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=False)

    # ================ 2. 模型加载 ================
    model_net = JacksonNet().to(device)
    print(f"Model total number of parameters: {count_parameters(model_net):,}")

    if os.path.exists(best_model_path):
        ckpt = torch.load(best_model_path, map_location=device)
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        incompat = model_net.load_state_dict(state_dict, strict=False)
        print(f"✅ 已加载模型参数: {best_model_path}")
        if incompat.missing_keys or incompat.unexpected_keys:
            print("⚠️ 参数不完全匹配:")
            print("Missing keys 示例:", incompat.missing_keys[:10], f"... 共{len(incompat.missing_keys)}")
            print("Unexpected keys 示例:", incompat.unexpected_keys[:10], f"... 共{len(incompat.unexpected_keys)}")
    else:
        raise FileNotFoundError(f"❌ 未找到模型文件: {best_model_path}")

    # —— 打印 smoother 学到的 alpha 系数 ——
    _print_smooth_alphas(model_net)

    model_net.eval()

    # ================ 3. 指标准备 ================
    ncc_loss_fn = NCCLoss()
    fn_mse = nn.MSELoss()

    # ---- baseline 累计（未经过模型/配准前）----
    baseline_total_ncc = 0.0
    baseline_total_mse = 0.0
    baseline_total_dice = 0.0
    baseline_total_hd95 = 0.0

    # ---- 模型后结果累计 ----
    total_ncc = 0.0
    total_mse = 0.0
    total_dice = 0.0
    total_hd95 = 0.0

    sample_count = 0
    hd95_calc = HausdorffDistance()
    STN = SpatialTransformer((256, 256)).to(device)

    # ================ 3.5 注册 TokenGate hooks（直方图累计） ================
    hist = GateHistograms(bins=50, out_dir="gate_vis")

    gate_modules = []
    for name, m in model_net.named_modules():
        if m.__class__.__name__ == "TokenGate":
            gate_modules.append((name, m))
    print(f"\n发现 {len(gate_modules)} 个 TokenGate：", [n for n, _ in gate_modules])

    hooks = []

    def make_hook(layer_name):
        def _hook(module, inputs, output):
            g = output
            if isinstance(g, (tuple, list)):
                if len(g) == 0:
                    return
                g = g[0]
            if not isinstance(g, torch.Tensor):
                return
            if g.dim() == 2:
                pass
            elif g.dim() == 3 and g.size(-1) == 1:
                g = g.squeeze(-1)
            else:
                return
            g = torch.clamp(g, 0.0, 1.0)
            hist.add_tensor(layer_name, g)
        return _hook

    for name, m in gate_modules:
        hooks.append(m.register_forward_hook(make_hook(name)))

    # ================ 4. 验证集推理 & 计算指标 ================
    with torch.no_grad():
        for data in val_loader:
            model1_img = data['model1'].to(device)
            model2_img = data['model2'].to(device)
            label = (data['label'].to(device) * 255.0)  # [B,1,H,W]
            seg = data['seg'].to(device)
            model1_normal_img = data['model_normal'].to(device)

            # =========================================================
            # 1) baseline：不经过模型，直接计算原始输入的四个指标
            # =========================================================
            # 若 seg 本身已经是 0/64/128/255，则改成 seg_baseline = seg
            seg_baseline = seg * 255.0

            baseline_ncc_batch = ncc_loss_fn(model1_img, model1_normal_img).item()
            baseline_mse_batch = fn_mse(model1_img, model1_normal_img).item()
            baseline_dice_batch = dice_coefficient(seg_baseline, label).item()
            baseline_hd95_vals = hd95_calc.compute(seg_baseline, label)
            baseline_hd95_batch = baseline_hd95_vals.mean().item()

            # =========================================================
            # 2) 模型输出：经过配准后再计算四个指标
            # =========================================================
            # 前向（会触发 hooks，累计 gate 直方图）
            flow_phi = model_net(model1_img, model2_img)
            model1_hat = STN(model1_img, flow_phi)

            seg_source_idx = label_value_to_index(seg * 255.0)
            seg_source_1h = label_to_one_hot(seg_source_idx, num_classes=4)
            seg_wrapped = STN(seg_source_1h, flow_phi)
            seg_idx = torch.argmax(seg_wrapped, dim=1)
            mapping = torch.tensor([0, 64.0, 128.0, 255.0], device=seg_idx.device)
            seg_discrete = mapping[seg_idx].unsqueeze(1)

            ncc_val_batch = ncc_loss_fn(model1_hat, model1_normal_img).item()
            mse_val_batch = fn_mse(model1_hat, model1_normal_img).item()
            dice_val_batch = dice_coefficient(seg_discrete, label).item()
            hd95_vals = hd95_calc.compute(seg_discrete, label)
            hd95_batch = hd95_vals.mean().item()

            bs = seg_discrete.size(0)

            # ---- baseline 累计 ----
            baseline_total_ncc += baseline_ncc_batch * bs
            baseline_total_mse += baseline_mse_batch * bs
            baseline_total_dice += baseline_dice_batch * bs
            baseline_total_hd95 += baseline_hd95_batch * bs

            # ---- 模型结果累计 ----
            total_ncc += ncc_val_batch * bs
            total_mse += mse_val_batch * bs
            total_dice += dice_val_batch * bs
            total_hd95 += hd95_batch * bs

            sample_count += bs

    # 移除 hooks
    for h in hooks:
        h.remove()

    # ================ 5. 计算平均指标 & 输出 ================
    baseline_avg_ncc = baseline_total_ncc / sample_count
    baseline_avg_mse = baseline_total_mse / sample_count
    baseline_avg_dice = baseline_total_dice / sample_count
    baseline_avg_hd95 = baseline_total_hd95 / sample_count

    avg_ncc = total_ncc / sample_count
    avg_mse = total_mse / sample_count
    avg_dice = total_dice / sample_count
    avg_hd95 = total_hd95 / sample_count

    print("========== 验证集指标 ==========")
    print("---- Baseline（未经过模型/配准前）----")
    print(f"NCC (loss形式) : {baseline_avg_ncc:.6f}")
    print(f"MSE           : {baseline_avg_mse:.6f}")
    print(f"Dice          : {baseline_avg_dice:.6f}")
    print(f"HD95          : {baseline_avg_hd95:.6f}")

    print("---- After Registration（经过模型后）----")
    print(f"NCC (loss形式) : {avg_ncc:.6f}")
    print(f"MSE           : {avg_mse:.6f}")
    print(f"Dice          : {avg_dice:.6f}")
    print(f"HD95          : {avg_hd95:.6f}")
    print("================================")

    # ================ 6. 保存每层直方图 PNG/CSV & 总结表 ================
    hist.save_all()
    print(f"✅ 已将每层 gate 的直方图与统计保存到：{os.path.abspath(hist.out_dir)}")
    print("   - 每层：.png 直方图、.csv 直方表（bin 左/右/计数）")
    print("   - 汇总：summary.csv（均值/标准差/近似分位数）")
