import math
import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


class DFTAttention(nn.Module):
    """
    频域通道注意力（带低/中/高频分桶的折中方案）
    - 使用 rfft2 振幅作为频域能量来源（数据驱动）
    - 轻量分桶（低/中/高）+ 可学习门控，增强高频保留能力
    - 参数量增幅极小，整体模型仍可控制在 <0.5M（以默认配置为准）
    """
    def __init__(self, channels: int):
        super().__init__()
        # 轻量频带门控：3 -> 3
        self.band_mlp = nn.Linear(3, 3, bias=True)
        # 原始通道注意力 MLP（保持与旧版同级别开销）
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )

        # 初始化频带门控为近似均衡（有利于稳定）
        nn.init.zeros_(self.band_mlp.weight)
        nn.init.zeros_(self.band_mlp.bias)

    @staticmethod
    def _make_freq_masks(h: int, w: int, device):
        """
        构造低/中/高频掩膜（径向分桶），形状 (3, h, w_rfft)
        w_rfft = w//2 + 1 for rfft2
        """
        fy = torch.fft.fftfreq(h, device=device)                # (h,)
        fx = torch.fft.rfftfreq(w, device=device)               # (w_rfft,)
        gy, gx = torch.meshgrid(fy, fx, indexing='ij')          # (h, w_rfft)

        radius = torch.sqrt(gx ** 2 + gy ** 2)                  # 归一化半径
        radius = radius / radius.max().clamp(min=1e-6)

        low  = (radius <= 0.25).float()
        mid  = ((radius > 0.25) & (radius <= 0.5)).float()
        high = (radius > 0.5).float()

        masks = torch.stack([low, mid, high], dim=0)            # (3, h, w_rfft)
        return masks

    def forward(self, x):
        """
        x: (B, C, H, W)
        返回: 通道加权后的特征 (B, C, H, W)
        """
        b, c, h, w = x.shape

        # 1) 频域振幅
        ffted = torch.fft.rfft2(x, norm='ortho')
        amplitude = torch.abs(ffted)  # (B, C, H, W//2+1)

        # 2) 低/中/高频分桶能量 (B, C, 3)
        masks = self._make_freq_masks(h, w, device=x.device).unsqueeze(0).unsqueeze(1)  # (1,1,3,H,W_r)
        energy_bands = (amplitude.unsqueeze(2) * masks).mean(dim=(-2, -1))              # (B, C, 3)

        # 3) 频带门控 (B, 3) -> softmax
        band_logits = self.band_mlp(energy_bands.mean(dim=1))  # 通道平均后得到全局频带能量
        band_gates = torch.softmax(band_logits, dim=-1)        # (B, 3)

        # 4) 加权汇聚到通道频域能量 (B, C)
        energy_mix = (energy_bands * band_gates.unsqueeze(1)).sum(dim=-1)

        # 5) 通道注意力
        y = self.fc(energy_mix).view(b, c, 1, 1)
        return x * y


class GateFreqIMDB(nn.Module):
    """基于傅里叶变换增强的信息多蒸馏模块（频域注意力折中版）"""
    def __init__(self, in_channels, distillation_rate=0.25):
        super().__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels

        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, 1)

        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.fusion = nn.Conv2d(self.distilled_channels * 4, in_channels, 1, 1, 0)

        # 采用折中版频域注意力
        self.fa = DFTAttention(in_channels)

    def forward(self, x):
        out_c1 = self.act(self.c1(x))
        d1, r1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c2 = self.act(self.c2(r1))
        d2, r2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c3 = self.act(self.c3(r2))
        d3, r3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)

        out_c4 = self.c4(r3)

        out = torch.cat([d1, d2, d3, out_c4], dim=1)
        out = self.fa(self.fusion(out))

        return out + x


@ARCH_REGISTRY.register()
class GateFreqIMDN(nn.Module):
    """
    GateFreqIMDN (DFT-ver, 带频带门控)
    支持多尺度 (x2, x3, x4) 且参数量 < 0.5M（默认 num_feat=48, num_blocks=6, distill=0.25）
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_blocks=6, scale=2):
        super().__init__()
        self.scale = scale

        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)

        self.blocks = nn.ModuleList([
            GateFreqIMDB(in_channels=num_feat) for _ in range(num_blocks)
        ])

        self.LR_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # --- 多尺度上采样 ---
        m_upsample = []
        if (scale & (scale - 1)) == 0:  # 2 or 4
            for _ in range(int(math.log(scale, 2))):
                m_upsample.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m_upsample.append(nn.PixelShuffle(2))
        elif scale == 3:
            m_upsample.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m_upsample.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'Scale {scale} 不受支持')

        m_upsample.append(nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))
        self.upsample = nn.Sequential(*m_upsample)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)

        fea = self.fea_conv(x)
        out = fea
        for block in self.blocks:
            out = block(out)

        out = self.LR_conv(out) + fea
        out = self.upsample(out)

        return out + base
