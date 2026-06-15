import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiSigmaGaussianSmoother(nn.Module):
    """
    在前向里对位移场做多σ高斯平滑的可学习正则层。
    支持 2D/3D，深度可分离卷积；α 用 softmax（可学习 or 固定），再用 λ∈[0,1] 控制与原场的混合。
    """
    def __init__(self,
                 channels: int,                 # 位移通道数：2(2D) or 3(3D)
                 sigmas=(0.5, 1.0, 2.0),        # 体素单位
                 alphas=None,                   # 与 sigmas 同长；None=均匀
                 learnable_alphas: bool = True,
                 dim: int = 2,                  # 2 or 3
                 mode: str = "lowpass",         # "lowpass" 或 "highfreq"
                 init_lambda: float = 0.5,      # 初始混合权
                 pad_mode: str = "reflect"):    # 反射填充更稳
        super().__init__()
        assert dim in (2,3)
        assert mode in ("lowpass", "highfreq")
        self.channels = channels
        self.sigmas = tuple(float(s) for s in sigmas)
        self.dim = dim
        self.mode = mode
        self.pad_mode = pad_mode

        # α 参数（softmax 归一化）
        S = len(self.sigmas)
        if alphas is None:
            init_alpha = torch.full((S,), 1.0 / S)
        else:
            assert len(alphas) == S
            a = torch.tensor(alphas, dtype=torch.float32)
            init_alpha = a / (a.sum() + 1e-8)
        if learnable_alphas:
            # 用 logits 表示，初始化到 init_alpha 的 log 空间
            self.alpha_logits = nn.Parameter(torch.log(init_alpha + 1e-8))
        else:
            self.register_buffer("alpha_fixed", init_alpha, persistent=False)
            self.alpha_logits = None

        # 预生成 1D 高斯核并注册为 buffer（不同 σ 不同半径）
        ks_1d = []
        for s in self.sigmas:
            rad = int(3.0)
            x = torch.arange(-rad, rad + 1, dtype=torch.float32)
            k = torch.exp(-0.5 * (x / s) ** 2)
            k = k / k.sum()
            ks_1d.append(k)
        # 存成 list of tensors；移动到 device/dtype 时会跟随 module.to()
        for i, k in enumerate(ks_1d):
            self.register_buffer(f"k1d_{i}", k, persistent=False)

    @staticmethod
    def _inv_sigmoid(y: float):
        y = min(max(y, 1e-6), 1-1e-6)
        return torch.log(torch.tensor(y/(1-y), dtype=torch.float32))

    def _alphas(self, device, dtype):
        if self.alpha_logits is not None:
            return torch.softmax(self.alpha_logits.to(device=device, dtype=dtype), dim=-1)
        else:
            return getattr(self, "alpha_fixed").to(device=device, dtype=dtype)

    def _sep_gauss_conv(self, u: torch.Tensor, k1d: torch.Tensor):
        """深度可分离高斯卷积：2D=先 y 再 x；3D= z→y→x。u:[B,C,H,W] 或 [B,C,D,H,W]"""
        C = u.size(1)
        if self.dim == 2:
            rad = k1d.numel() // 2
            # y 方向
            ky = k1d.view(1,1,-1,1).to(device=u.device, dtype=u.dtype).repeat(C,1,1,1)
            pad = (0,0, rad,rad)
            v = F.pad(u, pad, mode=self.pad_mode)
            v = F.conv2d(v, ky, padding=0, groups=C)
            # x 方向
            kx = k1d.view(1,1,1,-1).to(device=u.device, dtype=u.dtype).repeat(C,1,1,1)
            pad = (rad,rad, 0,0)
            v = F.pad(v, pad, mode=self.pad_mode)
            v = F.conv2d(v, kx, padding=0, groups=C)
            return v
        else:
            rad = k1d.numel() // 2
            # z
            kz = k1d.view(1,1,-1,1,1).to(device=u.device, dtype=u.dtype).repeat(C,1,1,1,1)
            pad = (0,0, 0,0, rad,rad)  # (W_left,W_right, H_top,H_bottom, D_front,D_back)
            v = F.pad(u, pad, mode=self.pad_mode)
            v = F.conv3d(v, kz, padding=0, groups=C)
            # y
            ky = k1d.view(1,1,1,-1,1).to(device=u.device, dtype=u.dtype).repeat(C,1,1,1,1)
            pad = (0,0, rad,rad, 0,0)
            v = F.pad(v, pad, mode=self.pad_mode)
            v = F.conv3d(v, ky, padding=0, groups=C)
            # x
            kx = k1d.view(1,1,1,1,-1).to(device=u.device, dtype=u.dtype).repeat(C,1,1,1,1)
            pad = (rad,rad, 0,0, 0,0)
            v = F.pad(v, pad, mode=self.pad_mode)
            v = F.conv3d(v, kx, padding=0, groups=C)
            return v

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: 位移/速度场，形状 2D [B, C(=2), H, W] 或 3D [B, C(=3), D, H, W]
        """
        assert u.size(1) == self.channels, f"channels mismatch: expected {self.channels}, got {u.size(1)}"
        alphas = self._alphas(u.device, u.dtype)  # [S]

        # 线性组合作为平滑基
        smooth_sum = 0.0
        for i, _ in enumerate(self.sigmas):
            k1d = getattr(self, f"k1d_{i}")
            u_s = self._sep_gauss_conv(u, k1d)
            smooth_sum = smooth_sum + alphas[i] * u_s

        out = smooth_sum


        return out
