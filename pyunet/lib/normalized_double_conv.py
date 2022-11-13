import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from depthwise_seperable_conv import DepthwiseSeperableConv

class NormalizedDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(NormalizedDoubleConv, self).__init__()

        self.first_conv = DepthwiseSeperableConv(in_channels, out_channels)
        self.first_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.first_actv = nn.PReLU(num_parameters=out_channels)

        self.second_conv = DepthwiseSeperableConv(out_channels, out_channels)
        self.second_norm = nn.InstanceNorm2d(out_channels, affine=True)
        self.second_actv = nn.PReLU(num_parameters=out_channels)

        self.skip_conn = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True)
        )

    def forward(self, x):
#        return self.conv(x)

        residual = x

        out = self.first_conv(x)
        out = self.first_norm(out)
        out = self.first_actv(out)

        out += self.skip_conn(residual)

        out = self.second_conv(out)
        out = self.second_norm(out)
        out = self.second_actv(out)

        out += self.skip_conn(residual)

        return out
