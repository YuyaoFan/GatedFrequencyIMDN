import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math

class DFTAttention(nn.Module):
    """
    真正的频域注意力模块 (Real DFT Attention)
    利用离散傅里叶变换 (RFFT2D) 分析特征图的频率成分
    """
    def __init__(self, channels):
        super(DFTAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        
        # 1. 将空域特征转换到频域 (Real FFT)
        # 返回结果维度: (b, c, h, w//2 + 1, 2) 其中 2 代表实部和虚部
        # 这里使用 rfft2 的 norm='ortho' 保证能量一致性
        ffted = torch.fft.rfft2(x, norm='ortho')
        
        # 2. 计算振幅 (Amplitude / Magnitude)
        # 振幅代表了该频率成分的能量强弱，也是高低频判断的核心依据
        amplitude = torch.abs(ffted)
        
        # 3. 提取频域特征表示
        # 我们将频域振幅图池化到 1x1，捕捉每个通道的全局频率分布能量
        y = self.avg_pool(amplitude).view(b, c)
        
        # 4. 生成注意力权重
        y = self.fc(y).view(b, c, 1, 1)
        
        return x * y

class FreqIMDB(nn.Module):
    """基于傅里叶变换增强的信息多蒸馏模块"""
    def __init__(self, in_channels, distillation_rate=0.25):
        super(FreqIMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate)
        self.remaining_channels = in_channels - self.distilled_channels
        
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, 1)
        
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.fusion = nn.Conv2d(self.distilled_channels * 4, in_channels, 1, 1, 0)
        
        # 核心：使用真正的 DFT 注意力
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
        # 融合后的特征通过 DFT 注意力进行高低频能量加权
        out = self.fa(self.fusion(out))
        
        return out + x

@ARCH_REGISTRY.register()
class FreqIMDN(nn.Module):
    """
    FreqIMDN (DFT-ver) 
    支持多尺度 (x2, x3, x4) 且参数量 < 0.5M
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_blocks=6, scale=2):
        super(FreqIMDN, self).__init__()
        self.scale = scale
        
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.blocks = nn.ModuleList([
            FreqIMDB(in_channels=num_feat) for _ in range(num_blocks)
        ])
        
        self.LR_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # --- 多尺度适配上采样逻辑 ---
        m_upsample = []
        if (scale & (scale - 1)) == 0:  # scale 2, 4
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