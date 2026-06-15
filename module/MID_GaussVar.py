# model/swin_with_embed.py
import torch
import torch.nn as nn
from module.SingleStage_CoAttentionGate import CoSingleStageSwin
import torch.nn.functional as F
from torch.distributions.normal import Normal
from module.RegularGauss import MultiSigmaGaussianSmoother


class SpatialTransformer(nn.Module):
    """
    Spatial Transformer，用于根据 flow 字段对输入图像进行形变。
    输入：
        - src: 形变前图像 [B, C, H, W]
        - flow: 位移场 [B, 2, H, W]，表示每个像素的 (dx, dy)
    输出：
        - warped image: [B, C, H, W]
    """
    def __init__(self, size):
        super().__init__()
        # 构造一个 meshgrid [H, W, 2]，用于表示图像坐标
        H, W = size
        grid_y, grid_x = torch.meshgrid(
            torch.arange(0, H), torch.arange(0, W), indexing='ij'
        )
        grid = torch.stack((grid_x, grid_y), dim=0)  # [2, H, W]
        self.register_buffer('grid', grid.float())  # 固定不更新参数

    def forward(self, src, flow):
        B, C, H, W = src.shape

        # 把 grid 放到 flow 的设备
        grid = self.grid.to(flow.device)

        # 计算采样点
        new_locs = grid[None, ...] + flow  # [B, 2, H, W]

        # 归一化到 [-1, 1]
        new_locs[:, 0, :, :] = 2.0 * (new_locs[:, 0, :, :] / (W - 1)) - 1.0
        new_locs[:, 1, :, :] = 2.0 * (new_locs[:, 1, :, :] / (H - 1)) - 1.0

        # [B, 2, H, W] → [B, H, W, 2]
        new_locs = new_locs.permute(0, 2, 3, 1)

        # ✅ 确保 new_locs 和 src 都在一个设备（通常是 GPU）
        new_locs = new_locs.to(src.device)

        warped = F.grid_sample(src, new_locs, align_corners=True, mode='bilinear', padding_mode='border')
        return warped


class PatchEmbed(nn.Module):
    def __init__(self,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 norm_layer: nn.Module = None,
                 num_modalities: int = 2,       # 新增：模态数
                 use_mod_token: bool = True):   # 新增：是否使用模态偏置
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else None

        self.use_mod_token = use_mod_token
        if use_mod_token:
            # 形状 [num_modalities, embed_dim] 的可学习模态向量表
            self.mod_embed = nn.Embedding(num_modalities, embed_dim)
            # 初始化为 0：起始不影响原有特征分布
            nn.init.zeros_(self.mod_embed.weight)

    def forward(self, x: torch.Tensor, mod_id=None) -> torch.Tensor:
        """
        x: [B, in_chans, H, W]
        mod_id: 可为 int（如 0/1）或 [B] 的 LongTensor，表示模态 ID。
        """
        x = self.proj(x)                                  # [B, embed_dim, H/ps, W/ps]
        x = x.flatten(2).transpose(1, 2).contiguous()     # [B, N, embed_dim]
        if self.norm:
            x = self.norm(x)                              # [B, N, embed_dim]

        # 加模态偏置（加在 LN 之后，避免被抹掉）
        if self.use_mod_token and (mod_id is not None):
            if isinstance(mod_id, int):
                mod_id = torch.full((x.size(0),), mod_id, dtype=torch.long, device=x.device)
            m = self.mod_embed(mod_id)                   # [B, embed_dim]
            x = x + m.unsqueeze(1)                       # [B, N, embed_dim]
        return x


class PoubleConv(nn.Module):
    """
    双卷积块：先将通道数从 in_channels 减半，再降到 2，
    仅包含 Conv2d + PReLU，无归一化
    """
    def __init__(self, in_channels: int):
        super().__init__()
        mid_channels = in_channels // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(mid_channels, 2, kernel_size=3, padding=1)
        self.prelu2 = nn.PReLU()

        self.conv2.weight = nn.Parameter(Normal(0, 1e-5).sample(self.conv2.weight.shape))
        self.conv2.bias = nn.Parameter(torch.zeros(self.conv2.bias.shape))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_double = self.double_conv(x)
        return x_double


class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)

        conv2d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv2d.weight.shape))
        conv2d.bias = nn.Parameter(torch.zeros(conv2d.bias.shape))

        super().__init__(conv2d)


class FeatureNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 下采样和上采样算子
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # 下采样后 conv
        self.conv1 = DoubleConv(1, 16, 16)
        self.conv11 = DoubleConv(16, 32, 32)
        self.conv12 = DoubleConv(32, 32, 32)
        self.conv13 = DoubleConv(32, 32, 32)
        self.conv14 = DoubleConv(32, 32, 32)
        # 跳跃连接后 conv，用 mid_ch + in_ch 作为输入通道

        self.conv2 = DoubleConv(16 + 32, 16, 16)
        self.conv21 = DoubleConv(32 + 32, 32, 32)
        self.conv22 = DoubleConv(32 + 32, 32, 32)
        self.conv23 = DoubleConv(32 + 32, 32, 32)

    def forward(self, I1):

        x1 = self.conv1(I1)
        x1p = self.pool(x1)

        x11 = self.conv11(x1p)
        x11p = self.pool(x11)

        x12 = self.conv12(x11p)
        x12p = self.pool(x12)

        x13 = self.conv13(x12p)
        x13p = self.pool(x13)

        x14 = self.conv14(x13p)

        x14u = self.upsample(x14)
        x14c = torch.cat([x14u, x13], dim=1)

        x23 = self.conv23(x14c)
        x23u = self.upsample(x23)
        x23c = torch.cat([x23u, x12], dim=1)

        x22 = self.conv22(x23c)
        x22u = self.upsample(x22)
        x22c = torch.cat([x22u, x11], dim=1)

        x21 = self.conv21(x22c)
        x21u = self.upsample(x21)
        x21c = torch.cat([x21u, x1], dim=1)

        x2 = self.conv2(x21c)

        return [x14, x23, x22, x21, x2]


class SwinWithEmbed(nn.Module):
    """
    将 PatchEmbed 和 CoSingleStageSwin 串起来；
    输入 x1, x2: [B, in_chans, img_size, img_size] → 输出 [B, 2, H, W] (位移场)
    """
    def __init__(self,
                 img_size: int = 256,
                 patch_size: int = 4,
                 in_chans: int = 3,
                 embed_dim: int = 96,
                 depth: int = 2,
                 num_heads: int = 8,
                 window_size: int = 8,
                 mlp_ratio: float = 4.,
                 mid1:int = 1,
                 end1:int = 1,
                 mid2:int = 1,
                 end2:int = 1,
                 mid3:int = 1,
                 end3:int = 1,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint: bool = False,
                 fused_window_process: bool = False):
        super().__init__()
        # 计算 patch 后的空间分辨率（token 网格大小）
        H = W = img_size // patch_size

        # 保存以便 reshape
        self.H = int(img_size)
        self.W = int(img_size)
        self.embed_dim = embed_dim
        self.patch_size = patch_size

        # 共享的 PatchEmbed（权重共享），仅此一份
        self.patch_embed = PatchEmbed(
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
            num_modalities=2,        # 两种模态：例如 T1=0, T2=1
            use_mod_token=True
        )

        # 你的跨模态 Swin stage（保持不变）
        self.swin = CoSingleStageSwin(
            input_resolution=(H, W),
            dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # ====== 下面保持你的原实现不变 ======
        in_channels = int(embed_dim / self.patch_size**2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.conv1 = DoubleConv((embed_dim + in_channels) * 2, mid1, end1)
        self.conv2 = DoubleConv(end1 + in_channels * 2, mid2, end2)
        self.conv3 = DoubleConv(end2 + in_channels * 2, mid3, end3)

        self.reg_head = RegistrationHead(
            in_channels=end3,
            out_channels=2,
            kernel_size=3,
        )

    def forward(self, x1, x2):
        # 保留原始输入分辨率支路
        xp0 = torch.cat([x1, x2], dim=1)   # [B, 2*C, H, W]

        # Patch Embedding（共享权重，但传不同的 mod_id）
        # 输出序列形状: [B, N, embed_dim]，N=(H/ps)*(W/ps)
        seq1 = self.patch_embed(x1, mod_id=0)   # 模态1
        seq2 = self.patch_embed(x2, mod_id=1)   # 模态2

        # Co-Swin：输入两个序列，输出两个序列（保持你的接口）
        x1, x2 = self.swin(seq1, seq2)          # [B, N, embed_dim]

        # 序列还原回 [B, C, H/ps, W/ps]
        B, N, embed_dim = x1.shape
        Hp = self.H // self.patch_size
        Wp = self.W // self.patch_size
        assert N == Hp * Wp, f"Token 数 {N} 不匹配网格 {Hp}×{Wp}"

        x1 = x1.view(B, Hp, Wp, embed_dim).permute(0, 3, 1, 2).contiguous()  # [B, embed_dim, Hp, Wp]
        x2 = x2.view(B, Hp, Wp, embed_dim).permute(0, 3, 1, 2).contiguous()
        x  = torch.cat([x1, x2], dim=1)  # [B, 2*embed_dim, Hp, Wp]

        # 你的后续金字塔 + 回归头（保持不变）
        xp1 = self.pool(xp0)
        xp2 = self.pool(xp1)

        xp2c  = torch.cat([x, xp2], dim=1)
        xp2cc = self.conv1(xp2c)
        xp2ccu = self.up(xp2cc)

        xp1c  = torch.cat([xp2ccu, xp1], dim=1)
        xp1cc = self.conv2(xp1c)
        xp1ccu = self.up(xp1cc)

        xp0c = torch.cat([xp1ccu, xp0], dim=1)
        x0   = self.conv3(xp0c)

        x = self.reg_head(x0)  # [B, 2, H, W]
        return x


class JacksonNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_net = FeatureNet()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        resolutions = [16, 32, 64, 128, 256]
        # 方法 1：用 ModuleList
        self.STNs = nn.ModuleList([
            SpatialTransformer((r, r))
            for r in resolutions
        ])
        in_chans_list = [32, 32, 32, 32, 16]
        embed_dims = [32 * 16, 32 * 16 , 32 * 16, 32 * 16, 16 * 16]
        window_sizes = [4, 8, 8, 8, 8]  # 对应每个分辨率的 window_size
        mid1_list = [512, 512, 512, 512, 256]
        end1_list = [256, 256, 256, 256, 128]
        mid2_list = [128, 128, 128, 128, 64]
        end2_list = [64, 64, 64, 64, 32]
        mid3_list = [32, 32, 32, 32, 16]
        end3_list = [16, 16, 16, 16, 8]
        self.STFs = nn.ModuleList([
            SwinWithEmbed(
                img_size=img_size,
                patch_size=4,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=2,
                num_heads=8,
                window_size=ws,
                mid1=md1,
                end1=ed1,
                mid2=md2,
                end2=ed2,
                mid3=md3,
                end3=ed3,
                norm_layer=nn.LayerNorm
            )
            for img_size, in_chans, embed_dim, ws, md1, ed1, md2, ed2, md3, ed3
            in zip(resolutions, in_chans_list, embed_dims, window_sizes, mid1_list,
                   end1_list, mid2_list, end2_list, mid3_list, end3_list)
        ])


        self.smooth = MultiSigmaGaussianSmoother(
            channels=2,
            sigmas=(0.5, 1.0, 2.0, 4.0),
            alphas=(0.4, 0.3, 0.2, 0.1),
            learnable_alphas=True,
            dim=2,
            mode="lowpass",
            init_lambda=0.5,
            pad_mode="reflect"
        )

    def forward(self, I1, I2):

        L1 = self.feature_net(I1)
        L2 = self.feature_net(I2)
        phi = torch.zeros((8,2,8,8),dtype=torch.float32,device=I1.device)

        for i in range(5):
            phi = self.up(phi)
            feature1 = self.STNs[i](L1[i], phi)
            feature2 = L2[i]
            phi = phi - self.STFs[i](feature1, feature2)
            # ★ 按你的要求：在这里做平滑
            phi = self.smooth(phi)

        return phi


'''if __name__ == "__main__":
    # 测试
    B, C, H_img, W_img = 8, 1, 256, 256
    x = torch.randn(B, C, H_img, W_img)
    y = torch.randn(B, C, H_img, W_img)
    model = JacksonNet()
    out = model(x,y)
    print("input :", x.shape)
    print("output:", out.shape)'''
