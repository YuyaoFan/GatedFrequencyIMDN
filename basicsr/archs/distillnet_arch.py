import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math

class BasicDistillBlock(nn.Module):
    """
    基础特征蒸馏块：只包含 Split 和 Concat
    不含 CCA 或 DFT 注意力，用于验证特征重用机制的有效性
    """
    def __init__(self, in_channels):
        super(BasicDistillBlock, self).__init__()
        self.dc = in_channels // 4 # 蒸馏通道数 12
        self.rc = in_channels - self.dc # 剩余通道数 36
        
        self.c1 = nn.Conv2d(in_channels, in_channels, 3, 1, 1)
        self.c2 = nn.Conv2d(self.rc, in_channels, 3, 1, 1)
        self.c3 = nn.Conv2d(self.rc, in_channels, 3, 1, 1)
        self.c4 = nn.Conv2d(self.rc, self.dc, 3, 1, 1)
        
        self.act = nn.LeakyReLU(0.05, inplace=True)
        self.fusion = nn.Conv2d(self.dc * 4, in_channels, 1, 1, 0)

    def forward(self, x):
        # 1. 逐步蒸馏特征
        out1 = self.act(self.c1(x))
        d1, r1 = torch.split(out1, (self.dc, self.rc), dim=1)
        
        out2 = self.act(self.c2(r1))
        d2, r2 = torch.split(out2, (self.dc, self.rc), dim=1)
        
        out3 = self.act(self.c3(r2))
        d3, r3 = torch.split(out3, (self.dc, self.rc), dim=1)
        
        out4 = self.c4(r3)
        
        # 2. 纯特征拼接重用 (无注意力)
        out = torch.cat([d1, d2, d3, out4], dim=1)
        out = self.fusion(out)
        
        return out + x

@ARCH_REGISTRY.register()
class BasicDistillNet(nn.Module):
    """
    演进实验 2：基础蒸馏网络
    逻辑：引入 IMDB 的 Split/Concat 逻辑，但去除注意力层，验证特征流设计的优越性
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_blocks=6, scale=2):
        super(BasicDistillNet, self).__init__()
        self.scale = scale
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        self.blocks = nn.ModuleList([
            BasicDistillBlock(in_channels=num_feat) for _ in range(num_blocks)
        ])
        
        self.LR_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        m_upsample = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m_upsample.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m_upsample.append(nn.PixelShuffle(2))
        elif scale == 3:
            m_upsample.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m_upsample.append(nn.PixelShuffle(3))
            
        m_upsample.append(nn.Conv2d(num_feat, num_out_ch, 3, 1, 1))
        self.upsample = nn.Sequential(*m_upsample)

    def forward(self, x):
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        fea = self.fea_conv(x)
        out = fea
        for block in self.blocks:
            out = block(out)
        out = self.LR_conv(out) + fea
        out = self.upsample(out)
        return out + base
