import torch
import torch.nn as nn
import math
from basicsr.utils.registry import ARCH_REGISTRY

class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

@ARCH_REGISTRY.register()
class SimpleCNN(nn.Module):
    '''
        SR network with a few convolutional layers and ReLU activations.
        input: 3-channel image
        output: feature map with same dimensions as input
    '''
    def __init__(self, scale=4):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.upsample = Upsample(scale, 3)

        
    def forward(self, x):
        x_input = x
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x)) + x_input
        x = self.upsample(x)
        return x
        
if __name__ == "__main__":
    # if you want to run the code, please comment line 25 @ARCH_REGISTRY.register()
    scale = 2
    model = SimpleCNN(scale)
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    print(output.shape)  # Expected output shape: (1, 3, 32 * scale, 32 * scale)
    params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {params / 1e6:.6f} (M)')
