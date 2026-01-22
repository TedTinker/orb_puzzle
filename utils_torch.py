import torch
from torch import nn
from torch.distributions import Normal
import torch.nn.functional as F
from kornia.color import rgb_to_hsv 



class Interpolate(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None, antialias=False):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.antialias = antialias

    def forward(self, x):
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
            antialias=self.antialias)
    
    
    
# CNN with capping (CC2d).
class ConstrainedConv2d(nn.Conv2d):
    def forward(self, input):
        return nn.functional.conv2d(
            input, self.weight, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def clamp_weights(self):
        self.weight.data.clamp_(-1.0, 1.0)
        if self.bias is not None:
            self.bias.data.clamp_(-1.0, 1.0)
            
            
            
# Convert RGB images to HSV images.
def rgb_to_circular_hsv(rgb):
    hsv_image = rgb_to_hsv(rgb) 
    hue = hsv_image[:, 0, :, :]
    hue_sin = (torch.sin(hue) + 1) / 2
    hue_cos = (torch.cos(hue) + 1) / 2
    hsv_circular = torch.stack([hue_sin, hue_cos, hsv_image[:, 1, :, :], hsv_image[:, 2, :, :]], dim=1)
    return hsv_circular