import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

def window_reverse(windows, window_size, H, W):
    """
    windows: (num_windows*B, window_size, window_size, C)
    return: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, H, W, -1)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1  = nn.Linear(in_features, hidden_features)
        self.act  = act_layer()
        self.fc2  = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def window_partition(x, window_size):
    """
    x: (B, H, W, C)
    return: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)

# ------------------------------------------------
# (新增) 逐 token 融合门：输入 concat([x_m, x_f, |x_m-x_f|]) → [B,N,1]
# ------------------------------------------------
class TokenGate(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, init_p: float = 0.5,temperature: float = 0.4):
        """
        dim: token 维度 C
        hidden_ratio: 隐层宽度比例
        init_p: 初始门值（粗层可设大些，细层可设小些；这里默认 0.5）
        """
        super().__init__()
        hidden = max(8, int(dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1)
        )
        # 让初始门约等于 init_p
        nn.init.constant_(self.net[-1].bias, math.log(init_p / (1 - init_p)))
        self.temperature = temperature
    def forward(self, xm: torch.Tensor, xf: torch.Tensor) -> torch.Tensor:
        # xm, xf: [B, N, C]
        g_in = torch.cat([xm, xf, (xm - xf).abs()], dim=-1)  # [B, N, 3C]
        g = torch.sigmoid(self.net(g_in) / self.temperature)        # [B, N, 1]
        return g

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim         = dim
        self.window_size = window_size
        self.num_heads   = num_heads
        head_dim = dim // num_heads
        self.scale  = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords   = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_f = torch.flatten(coords, 1)
        relative_coords = coords_f[:, :, None] - coords_f[:, None, :]  # 2,H*W,H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W,H*W,2
        relative_coords[:, :, 0] += window_size[0] - 1
        relative_coords[:, :, 1] += window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer('relative_position_index', relative_coords.sum(-1))  # H*W,H*W

        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)  # ← 将在 Block 里被共享层覆盖
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj      = nn.Linear(dim, dim)                     # ← 同上
        self.proj_drop = nn.Dropout(proj_drop)
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N, N, -1).permute(2, 0, 1)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# ------------------------------------------------
# 2) CrossWindowAttention: Q from one modality, KV from the other
#    （结构保持，但其内部 WindowAttention 的 qkv/proj 将被 Block 注入共享层）
# ------------------------------------------------
class CrossWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.self_att = WindowAttention(dim, window_size, num_heads,
                                       qkv_bias, qk_scale,
                                       attn_drop, proj_drop)
    def forward(self, q_x, kv_x, mask=None):
        # q_x, kv_x: [B_*nW, N, C]
        B_, N, C = q_x.shape

        # 1) 统一计算 Q/K/V 的线性映射（由共享 qkv 层完成）
        qkv_q = (
            self.self_att.qkv(q_x)
            .reshape(B_, N, 3, self.self_att.num_heads, C // self.self_att.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q = qkv_q[0]  # [B_, num_heads, N, head_dim]

        qkv_kv = (
            self.self_att.qkv(kv_x)
            .reshape(B_, N, 3, self.self_att.num_heads, C // self.self_att.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        k = qkv_kv[1]  # [B_, num_heads, N, head_dim]
        v = qkv_kv[2]  # [B_, num_heads, N, head_dim]

        # 2) scaled dot-product attention
        q = q * self.self_att.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        # 3) 相对位置偏置
        rel = self.self_att.relative_position_bias_table[
            self.self_att.relative_position_index.view(-1)
        ]
        rel = rel.view(N, N, -1).permute(2, 0, 1)  # [num_heads, N, N]
        attn = attn + rel.unsqueeze(0)             # [B_, num_heads, N, N]

        # 4) 可选窗口 mask
        if mask is not None:
            nW = mask.shape[0]
            attn = (
                attn.view(B_ // nW, nW, self.self_att.num_heads, N, N)
                + mask.unsqueeze(1).unsqueeze(0)
            ).view(-1, self.self_att.num_heads, N, N)

        # 5) softmax, dropout
        attn = self.self_att.softmax(attn)
        attn = self.self_att.attn_drop(attn)

        # 6) 输出映射
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.self_att.proj(x)        # ← 将使用共享 proj
        x = self.self_att.proj_drop(x)
        return x

# ------------------------------------------------
# 3) CoSwinTransformerBlock：加入逐 token 融合门 + ★在本 Block 内共享 Wq/Wk/Wv/Proj
# ------------------------------------------------
class CoSwinTransformerBlock(nn.Module):
    def __init__(self,
                 dim: int,
                 input_resolution: tuple,
                 num_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 mlp_ratio: float = 4.,
                 qkv_bias: bool = True,
                 qk_scale = None,
                 drop: float = 0.,
                 attn_drop: float = 0.,
                 drop_path: float = 0.,
                 norm_layer = nn.LayerNorm,
                 p_init = 0.5):
        super().__init__()

        self.input_resolution = input_resolution
        self.window_size      = window_size
        self.shift_size       = shift_size

        # self-attn（两路）
        self.norm1_m   = norm_layer(dim)
        self.norm1_f   = norm_layer(dim)
        self.self_att = WindowAttention(
            dim, to_2tuple(window_size), num_heads,
            qkv_bias, qk_scale, attn_drop, drop)


        # self-attn 后 MLP
        self.norm_sa_m = norm_layer(dim)
        self.norm_sa_f = norm_layer(dim)
        self.mlp_sa_m  = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.mlp_sa_f  = Mlp(dim, int(dim * mlp_ratio), drop=drop)

        # cross-attn
        self.cross_att = CrossWindowAttention(
            dim, to_2tuple(window_size), num_heads,
            qkv_bias, qk_scale, attn_drop, drop)

        # ★★★ 关键：定义“共享”的 QKV 与 Proj，并注入到三处注意力 ★★★
        self.qkv_shared  = nn.Linear(dim, dim * 3, bias=qkv_bias)

        for att in (self.self_att, self.cross_att.self_att):
            att.qkv  = self.qkv_shared


        # 逐 token 融合门
        self.gate_m2f = TokenGate(dim, hidden_ratio=0.25, init_p=p_init)
        self.gate_f2m = TokenGate(dim, hidden_ratio=0.25, init_p=p_init)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # cross-attn 后 MLP
        self.norm2_m = norm_layer(dim)
        self.norm2_f = norm_layer(dim)
        self.mlp_m   = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.mlp_f   = Mlp(dim, int(dim * mlp_ratio), drop=drop)
        self.norm3_m = norm_layer(dim)
        self.norm3_f = norm_layer(dim)

        # SW-MSA 的 mask
        if shift_size > 0:
            H, W = input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)) \
                                   .masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_m: torch.Tensor, x_f: torch.Tensor):
        B, N, C = x_m.shape
        H, W    = self.input_resolution

        # -------- self-attention (moving) --------
        m = self.norm1_m(x_m)
        m = m.view(B, H, W, C)
        if self.shift_size > 0:
            m = torch.roll(m, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        win_m  = window_partition(m, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_m = self.self_att(win_m, mask=self.attn_mask)
        m      = window_reverse(attn_m.view(-1, self.window_size, self.window_size, C),
                                 self.window_size, H, W)
        if self.shift_size > 0:
            m = torch.roll(m, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        m   = m.view(B, N, C)
        x_m = x_m + self.drop_path(m)
        xm_sa = self.norm_sa_m(x_m)
        x_m   = x_m + self.drop_path(self.mlp_sa_m(xm_sa))

        # -------- self-attention (fixed) --------
        f = self.norm1_f(x_f)
        f = f.view(B, H, W, C)
        if self.shift_size > 0:
            f = torch.roll(f, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        win_f  = window_partition(f, self.window_size).view(-1, self.window_size * self.window_size, C)
        attn_f = self.self_att(win_f, mask=self.attn_mask)
        f      = window_reverse(attn_f.view(-1, self.window_size, self.window_size, C),
                                 self.window_size, H, W)
        if self.shift_size > 0:
            f = torch.roll(f, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        f   = f.view(B, N, C)
        x_f = x_f + self.drop_path(f)
        xf_sa = self.norm_sa_f(x_f)
        x_f   = x_f + self.drop_path(self.mlp_sa_f(xf_sa))

        # -------- cross-attention m↔f --------
        m2 = self.norm2_m(x_m)  # [B,N,C]
        f2 = self.norm2_f(x_f)  # [B,N,C]

        m2w = m2.view(B, H, W, C)
        f2w = f2.view(B, H, W, C)
        if self.shift_size > 0:
            m2w = torch.roll(m2w, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            f2w = torch.roll(f2w, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        wm = window_partition(m2w, self.window_size).view(-1, self.window_size * self.window_size, C)
        wf = window_partition(f2w, self.window_size).view(-1, self.window_size * self.window_size, C)

        cm = self.cross_att(wm, wf, self.attn_mask)  # m attends to f
        cf = self.cross_att(wf, wm, self.attn_mask)  # f attends to m

        cm = window_reverse(cm.view(-1, self.window_size, self.window_size, C),
                            self.window_size, H, W).view(B, N, C)
        cf = window_reverse(cf.view(-1, self.window_size, self.window_size, C),
                            self.window_size, H, W).view(B, N, C)

        ''' # ★ 逐 token 融合门（不改 attn，仅控制融合强度）
        g_m2f = self.gate_m2f(x_m, x_f)  # [B,N,1]
        g_f2m = self.gate_f2m(x_f, x_m)  # [B,N,1]

        x_m = x_m + self.drop_path(g_m2f * cm)  # 广播到通道维
        x_f = x_f + self.drop_path(g_f2m * cf)'''

        x_m = x_m + self.drop_path(cm)  # 广播到通道维
        x_f = x_f + self.drop_path(cf)

        # -------- MLP after cross-attn --------
        xm  = self.norm3_m(x_m)
        xf  = self.norm3_f(x_f)
        x_m = x_m + self.drop_path(self.mlp_m(xm))
        x_f = x_f + self.drop_path(self.mlp_f(xf))

        return x_m, x_f


class BasicCoLayer(nn.Module):
    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        if isinstance(drop_path, float):
            drop_path=[drop_path]*depth
        p_list = [0.9,0.8,0.7,0.6,0.6]
        self.blocks = nn.ModuleList([
            CoSwinTransformerBlock(
                dim, input_resolution, num_heads,
                window_size, shift_size=0 if i%2==0 else window_size//2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop, attn_drop=attn_drop, drop_path=drop_path[i],
                norm_layer=norm_layer, p_init=p_list[i]
            ) for i in range(depth)
        ])
    def forward(self, x_m, x_f):
        for blk in self.blocks:
            x_m, x_f = blk(x_m, x_f)
        return x_m, x_f

# ------------------------------------------------
# 5) CoSingleStageSwin：接受两路输入 [B,N,C]
# ------------------------------------------------
class CoSingleStageSwin(nn.Module):
    def __init__(self, input_resolution=(56,56), dim=96, depth=2, num_heads=3, window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, use_checkpoint=False):
        super().__init__()
        self.co_layer = BasicCoLayer(
            dim, input_resolution, depth, num_heads, window_size,
            mlp_ratio, qkv_bias, qk_scale,
            drop_rate, attn_drop_rate, drop_path_rate,
            norm_layer, use_checkpoint
        )
        self.norm = norm_layer(dim)
    def forward(self, xm, xf):
        xm, xf = self.co_layer(xm, xf)

        return self.norm(xm), self.norm(xf)
