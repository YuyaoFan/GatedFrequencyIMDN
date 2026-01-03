import torch
from torch import nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY
import math

class ResidualBlock(nn.Module):
    """
    标准的残差块：由两个 3x3 卷积组成
    用于验证增加深度对性能的影响
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        res = self.relu(self.conv1(x))
        res = self.conv2(res)
        return x + res

@ARCH_REGISTRY.register()
class DeepResNet(nn.Module):
    """
    演进实验 1：深度残差网络
    逻辑：在 SimpleCNN 基础上增加 Block 数量并添加全局残差
    """
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=48, num_blocks=6, scale=2):
        super(DeepResNet, self).__init__()
        self.scale = scale
        
        # 浅层特征提取
        self.fea_conv = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        
        # 堆叠残差块 (与最终模型 Block 数对齐)
        self.blocks = nn.ModuleList([
            ResidualBlock(channels=num_feat) for _ in range(num_blocks)
        ])
        
        self.LR_conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        
        # 上采样逻辑 (保持对齐)
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
        # 线性插值路径
        base = F.interpolate(x, scale_factor=self.scale, mode='bilinear', align_corners=False)
        
        fea = self.fea_conv(x)
        out = fea
        for block in self.blocks:
            out = block(out)
        
        out = self.LR_conv(out) + fea
        out = self.upsample(out)
        
        return out + base
