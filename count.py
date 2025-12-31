from basicsr.archs.FreqIMDN_arch import FreqIMDN
import torch
model = FreqIMDN(num_feat=48, num_blocks=6)
params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {params / 1e6:.4f}M")