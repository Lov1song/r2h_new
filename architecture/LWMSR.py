import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
import numpy as np
import math
from functools import partial
from einops import repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        output = x.div(keep_prob) * random_tensor
        return output

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=2.,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 输入投影
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # 深度可分离卷积
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs
        )
        self.act = nn.SiLU()

        # 四个方向的 x 投影
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # 四个方向的 dt 投影
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # A 和 D 参数
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # B,4,D,L
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0] + inv_y[:, 0] + wh_y + invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        # expects x: B, H, W, C
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: callable = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            mlp_ratio: float = 2.,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(d_model=hidden_dim, d_state=d_state,
                                   expand=mlp_ratio, dropout=attn_drop_rate, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, input):
        # input: B C H W -> permute to B H W C for SS2D
        input = input.permute(0, 2, 3, 1)          # B H W C
        B, H, W, C = input.shape
        x = self.ln_1(input)
        x = self.drop_path(self.self_attention(x))
        x = x.view(B, H, W, C).permute(0, 3, 1, 2) # B C H W
        return x

# -------------------------
# Efficient DWT / IDWT（保留并稍微整理）
# -------------------------
class EfficientDWT(nn.Module):
    """高效离散小波变换层（Haar 优化）"""
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = wave
        
        # Haar小波核: shape (4, 1, 2, 2)
        if wave == 'haar':
            self.register_buffer('kernel', torch.tensor([
                [[[1.0, 1.0], [1.0, 1.0]]],   # LL
                [[[1.0, -1.0], [1.0, -1.0]]], # LH
                [[[1.0, 1.0], [-1.0, -1.0]]], # HL
                [[[1.0, -1.0], [-1.0, 1.0]]]  # HH
            ]) * 0.5, persistent=False)

    def forward(self, x):
        """x: [B, C, H, W] 返回: [B, 4*C, H//2, W//2]"""
        batch_size, channels, height, width = x.shape
        
        if self.wave == 'haar':
            expanded_kernel = self.kernel.repeat(channels, 1, 1, 1)  # (4*channels,1,2,2)
            # conv2d: in_channels=channels, groups=channels -> kernel must be (channels*4,1,2,2)
            coeffs = F.conv2d(x, expanded_kernel, stride=2, groups=channels)
            return coeffs
        else:
            return self._fallback_dwt(x)

    def _fallback_dwt(self, x):
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size * channels, 1, height, width)
        
        x_np = x.detach().cpu().numpy()
        coeffs_list = []
        
        for i in range(x_np.shape[0]):
            coeffs = pywt.dwt2(x_np[i, 0], self.wave)
            LL, (LH, HL, HH) = coeffs
            coeffs_combined = np.stack([LL, LH, HL, HH], axis=0)
            coeffs_list.append(coeffs_combined)
        
        coeffs_tensor = torch.tensor(np.array(coeffs_list), dtype=x.dtype, device=x.device)
        return coeffs_tensor.reshape(batch_size, channels * 4, height // 2, width // 2)

class EfficientIDWT(nn.Module):
    """高效逆离散小波变换层（Haar 优化）"""
    def __init__(self, wave='haar'):
        super().__init__()
        self.wave = wave
        
        if wave == 'haar':
            self.register_buffer('kernel', torch.tensor([
                [[[1.0, 1.0], [1.0, 1.0]]],
                [[[1.0, 1.0], [-1.0, -1.0]]],
                [[[1.0, -1.0], [1.0, -1.0]]],
                [[[1.0, -1.0], [-1.0, 1.0]]]
            ]) * 0.5, persistent=False)

    def forward(self, x):
        """x: [B, 4*C, H, W] 返回: [B, C, 2*H, 2*W]"""
        if self.wave == 'haar':
            return self._haar_idwt_fast(x)
        else:
            return self._fallback_idwt(x)

    def _haar_idwt_fast(self, x):
        batch_size, total_channels, height, width = x.shape
        assert total_channels % 4 == 0, "Total channels must be divisible by 4"
        orig_channels = total_channels // 4
        
        expanded_kernel = self.kernel.repeat(orig_channels, 1, 1, 1)
        reconstructed = F.conv_transpose2d(
            x, expanded_kernel, 
            stride=2, groups=orig_channels
        )
        
        return reconstructed

    def _fallback_idwt(self, x):
        batch_size, total_channels, height, width = x.shape
        assert total_channels % 4 == 0, "Channels must be divisible by 4"
        orig_channels = total_channels // 4
        
        x = x.reshape(batch_size * orig_channels, 4, height, width)
        x_np = x.detach().cpu().numpy()
        reconstructed_list = []
        
        for i in range(x_np.shape[0]):
            coeffs = (x_np[i, 0], (x_np[i, 1], x_np[i, 2], x_np[i, 3]))
            reconstructed = pywt.idwt2(coeffs, self.wave)
            reconstructed_list.append(reconstructed)
        
        reconstructed_tensor = torch.tensor(np.array(reconstructed_list), dtype=x.dtype, device=x.device)
        return reconstructed_tensor.reshape(batch_size, orig_channels, height * 2, width * 2)


# WavePath：把 LL 用 SS2D（Mamba）做全局建模，LH/HL/HH 用卷积局部处理

class WavePath(nn.Module):
    def __init__(self, in_c, hidden_c, out_c, d_state=16, expand=2):
        super().__init__()
        self.dwt = EfficientDWT()
        self.idwt = EfficientIDWT()

        # 低频 LL: 投影 -> VSSBlock
        self.ll_proj = nn.Conv2d(in_c, hidden_c, 1)
        self.ll_vss = VSSBlock(hidden_dim=hidden_c, d_state=d_state)


        # 高频三子带: 使用轻量卷积分支（局部模型）
        def make_hf_branch(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                nn.GELU(),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.GELU()
            )

        #我们对每个高频分量先用 1x1 投影到 hidden_c，再卷积
        self.hf_proj = nn.Conv2d(in_c, hidden_c, 1)
        self.lh_cnn = make_hf_branch(hidden_c, hidden_c)
        self.hl_cnn = make_hf_branch(hidden_c, hidden_c)
        self.hh_cnn = make_hf_branch(hidden_c, hidden_c)

        # 融合投影：4*hidden -> out_c（注意通道数）
        self.post = nn.Conv2d(4 * hidden_c, out_c, 1)

    def forward(self, x):
        """
        x: B, C, H, W
        returns: B, out_c, H, W  (与原接口一致)
        """
        B, C, H, W = x.shape
        dwt = self.dwt(x)  # B, 4*C, H//2, W//2
        ll, lh, hl, hh = torch.chunk(dwt, 4, dim=1)  # 每个: B, C, H/2, W/2

        # 低频部分
        ll = self.ll_proj(ll)
        ll = self.ll_vss(ll)  # 直接输入 B,C,H,W，不用再 permute


        # 高频用小卷积网络（先投影）
        lh = self.hf_proj(lh); lh = self.lh_cnn(lh)
        hl = self.hf_proj(hl); hl = self.hl_cnn(hl)
        hh = self.hf_proj(hh); hh = self.hh_cnn(hh)

        # 融合并投影回 out_c，然后逆变换回原尺度
        fused = torch.cat([ll, lh, hl, hh], dim=1)  # B, 4*hidden_c, H/2, W/2
        fused = self.post(fused)  # B, out_c, H/2, W/2

        # IDWT 恢复到 H,W
        return self.idwt(fused)  # B, out_c, H, W

# Multi-Scale Edge Attention Module (MSEAM)
class MSEAM(nn.Module):

    def __init__(self, in_c):
        super().__init__()
        # ---- Sobel 核 ----
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32)
        sobel_y = sobel_x.t()
        self.register_buffer("sobel_x", sobel_x[None, None, :, :])
        self.register_buffer("sobel_y", sobel_y[None, None, :, :])

        self.mscale = nn.ModuleList([
            nn.Conv2d(1, 1, k, padding=k // 2, bias=False)
            for k in [3, 5, 7]
        ])
        self.fuse = nn.Conv2d(3, 1, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')

        #Sobel
        gx = F.conv2d(x_pad, self.sobel_x.repeat(C, 1, 1, 1), groups=C)
        gy = F.conv2d(x_pad, self.sobel_y.repeat(C, 1, 1, 1), groups=C)
        edge = (torch.abs(gx) + torch.abs(gy)).mean(1, keepdim=True)  # B,1,H,W

        edge_feats = [conv(edge) for conv in self.mscale]
        edge_fused = self.sigmoid(self.fuse(torch.cat(edge_feats, 1))) 

        out = x * (1 + edge_fused)
        return out

class SpectralPath(nn.Module):
    def __init__(self, in_c=3, out_c=64, mid_c=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_c, mid_c, 3, 1, 1),
            nn.GELU(),
            VSSBlock(hidden_dim=mid_c, d_state=16),
            nn.Conv2d(mid_c, out_c, 3, 1, 1)
        )

    def forward(self, x):
        return self.body(x)

class LateFusion(nn.Module):
    def __init__(self, c_list, out_band=31):
        """
        c_list: 列表，表示每个分支的通道数，例如 [wave_ch, spatial_ch, spectral_ch]
        out_band: 最终输出光谱波段数
        """
        super().__init__()
        total = sum(c_list)
        self.fuse = nn.Sequential(
            nn.Conv2d(total, max(total // 2, 1), 1),
            nn.GELU(),
            nn.Conv2d(max(total // 2, 1), out_band, 1)
        )

    def forward(self, *feats):
        return self.fuse(torch.cat(feats, 1))


class LWMSR(nn.Module):
    """
    Lightweight Wavelet-Mamba Spectral Reconstruction.
    in_channels : RGB input channels (default 3)
    out_channels: number of HSI bands to reconstruct (e.g. 150 for band 10-159)
    n_feat      : internal feature width (default 64)
    """
    def __init__(self, in_channels=3, out_channels=150, n_feat=64):
        super().__init__()
        c = n_feat
        spectral_c = max(out_channels // 2, c)  # spectral path mid-size

        self.wave_path    = WavePath(in_channels, c, c)
        self.spatial_path = MSEAM(in_c=in_channels)
        self.spectral_path = SpectralPath(in_c=in_channels, out_c=spectral_c, mid_c=c)

        # wave_path returns c//4 channels after IDWT
        self.fusion = LateFusion([c // 4, in_channels, spectral_c], out_channels)

    def forward(self, x):
        f_w = self.wave_path(x)      # B, c//4, H, W
        f_s = self.spatial_path(x)   # B, in_channels, H, W
        f_p = self.spectral_path(x)  # B, spectral_c, H, W
        return self.fusion(f_w, f_s, f_p)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = LWMSR(in_channels=3, out_channels=150, n_feat=64).to(device)
    rgb = torch.randn(1, 3, 128, 128).to(device)
    with torch.no_grad():
        hsi = net(rgb)
    print('Output shape:', hsi.shape)  # (1, 150, 128, 128)
    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f'Trainable params: {n_params / 1e6:.2f}M')
