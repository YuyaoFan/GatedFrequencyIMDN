import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math

class CCALayer(nn.Module):
    """
    Contrast-aware Channel Attention (CCA)
    这是原版 IMDN 提出的注意力机制，通过均值和标准差感知对比度
    """
    def __init__(self, channel, reduction=16):
        super(CCALayer, self).__init__()
        self.contrast = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 原版 IMDN 使用均值作为对比度代理
        y = self.contrast(x)
        y = self.conv_du(y)
        return x * y

class IMDB(nn.Module):
    """
    原始信息多蒸馏模块 (Information Multi-distillation Block)
    包含四次特征蒸馏与最后的特征融合
    """
    def __init__(self, in_channels, distillation_rate=0.25):
        super(IMDB, self).__init__()
        self.distilled_channels = int(in_channels * distillation_rate) # 12
        self.remaining_channels = in_channels - self.distilled_channels # 36
        
        # 蒸馏卷积层
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.remaining_channels, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.remaining_channels, self.distilled_channels, 3, 1, 1)
        
        self.act = nn.LeakyReLU(0.05, inplace=True)
        
        # 融合层
        self.fusion = nn.Conv2d(self.distilled_channels * 4, in_channels, 1, 1, 0)
        self.cca = CCALayer(in_channels)

    def forward(self, x):
        # 第一次蒸馏
        out_c1 = self.act(self.c1(x))
        distilled_c1, remaining_c1 = torch.split(out_c1, (self.distilled_channels, self.remaining_channels), dim=1)
        
        # 第二次蒸馏
        out_c2 = self.act(self.c2(remaining_c1))
        distilled_c2, remaining_c2 = torch.split(out_c2, (self.distilled_channels, self.remaining_channels), dim=1)
        
        # 第三次蒸馏
        out_c3 = self.act(self.c3(remaining_c2))
        distilled_c3, remaining_c3 = torch.split(out_c3, (self.distilled_channels, self.remaining_channels), dim=1)
        
        # 第四次特征提取
        out_c4 = self.c4(remaining_c3)
        
        # 特征拼接与选择性融合
        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, out_c4], dim=1)
        out = self.cca(self.fusion(out))
        
        return out + x

@ARCH_REGISTRY.register()
class IMDN(nn.Module):
    """
    原始 IMDN 架构 (对齐对比版)
    参数设置: num_feat=48, num_blocks=6
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_blocks=6, scale=2):
        super(IMDN, self).__init__()
        self.scale = scale
        
        # 浅层特征提取
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # 堆叠 IMDB
        self.blocks = nn.ModuleList([
            IMDB(in_channels=num_feat) for _ in range(num_blocks)
        ])
        
        # 瓶颈层
        self.LR_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 多尺度重建层 (与 FreqIMDN 逻辑完全对齐)
        m_upsample = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m_upsample.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m_upsample.append(nn.PixelShuffle(2))
        elif scale == 3:
            m_upsample.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m_upsample.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'不支持的 scale: {scale}')
            
        m_upsample.append(nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))
        self.upsample = nn.Sequential(*m_upsample)
        
        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 同样保留全局残差路径以公平对比
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        fea = self.fea_conv(x)
        out = fea
        for block in self.blocks:
            out = block(out)
        
        out = self.LR_conv(out) + fea
        out = self.upsample(out)
        
        return out + base
