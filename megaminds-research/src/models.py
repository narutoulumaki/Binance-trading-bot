"""
YOLOv7 Model Architecture Implementation
Simplified implementation for Pascal VOC object detection
"""

import torch
import torch.nn as nn
import math
from typing import List, Tuple


def autopad(k, p=None):
    """Auto-padding to maintain spatial dimensions."""
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation."""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck block."""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast."""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Detect(nn.Module):
    """YOLOv7 Detect head for object detection."""
    stride = None  # strides computed during build
    
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)
        self.inplace = True

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device
        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()
        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand((1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class YOLOv7(nn.Module):
    """YOLOv7 model architecture."""
    def __init__(self, nc=20, anchors=None, ch=3, width_multiple=1.0, depth_multiple=1.0):
        super().__init__()
        self.nc = nc
        
        # Default anchors for 3 detection layers
        if anchors is None:
            anchors = [
                [12, 16, 19, 36, 40, 28],  # P3/8
                [36, 75, 76, 55, 72, 146],  # P4/16
                [142, 110, 192, 243, 459, 401]  # P5/32
            ]
        
        def make_divisible(x, divisor=8):
            return math.ceil(x / divisor) * divisor
        
        def scale_width(channels):
            return make_divisible(channels * width_multiple)
        
        def scale_depth(number):
            return max(round(number * depth_multiple), 1)
        
        # Backbone
        self.backbone = nn.ModuleList([
            Conv(ch, scale_width(32), 3, 2),  # 0-P1/2
            Conv(scale_width(32), scale_width(64), 3, 2),  # 1-P2/4
            C3(scale_width(64), scale_width(64), scale_depth(1)),  # 2
            Conv(scale_width(64), scale_width(128), 3, 2),  # 3-P3/8
            C3(scale_width(128), scale_width(128), scale_depth(2)),  # 4
            Conv(scale_width(128), scale_width(256), 3, 2),  # 5-P4/16
            C3(scale_width(256), scale_width(256), scale_depth(3)),  # 6
            Conv(scale_width(256), scale_width(512), 3, 2),  # 7-P5/32
            C3(scale_width(512), scale_width(512), scale_depth(1)),  # 8
            SPPF(scale_width(512), scale_width(512), k=5),  # 9
        ])
        
        # Head
        self.head = nn.ModuleList([
            Conv(scale_width(512), scale_width(256), 1, 1),  # 10
            nn.Upsample(None, 2, 'nearest'),  # 11
            C3(scale_width(512), scale_width(256), scale_depth(1), shortcut=False),  # 12
            Conv(scale_width(256), scale_width(128), 1, 1),  # 13
            nn.Upsample(None, 2, 'nearest'),  # 14
            C3(scale_width(256), scale_width(128), scale_depth(1), shortcut=False),  # 15 (P3/8-small)
            Conv(scale_width(128), scale_width(128), 3, 2),  # 16
            C3(scale_width(256), scale_width(256), scale_depth(1), shortcut=False),  # 17 (P4/16-medium)
            Conv(scale_width(256), scale_width(256), 3, 2),  # 18
            C3(scale_width(512), scale_width(512), scale_depth(1), shortcut=False),  # 19 (P5/32-large)
        ])
        
        # Detect
        self.detect = Detect(
            nc=nc,
            anchors=anchors,
            ch=(scale_width(128), scale_width(256), scale_width(512))
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass through the network."""
        # Backbone
        x = self.backbone[0](x)  # P1
        x = self.backbone[1](x)  # P2
        x = self.backbone[2](x)
        x = self.backbone[3](x)  # P3
        x3 = self.backbone[4](x)  # Save for concatenation
        x = self.backbone[5](x3)  # P4
        x4 = self.backbone[6](x)  # Save for concatenation
        x = self.backbone[7](x4)  # P5
        x = self.backbone[8](x)
        x5 = self.backbone[9](x)  # Save for detection
        
        # Head
        x = self.head[0](x5)  # Conv
        x = self.head[1](x)  # Upsample
        x = torch.cat([x, x4], 1)  # Concatenate with P4
        x = self.head[2](x)  # C3
        x = self.head[3](x)  # Conv
        x_p3 = self.head[4](x)  # Upsample
        x_p3 = torch.cat([x_p3, x3], 1)  # Concatenate with P3
        x_p3 = self.head[5](x_p3)  # C3 -> P3 output
        
        x_p4 = self.head[6](x_p3)  # Downsample
        x_p4 = torch.cat([x_p4, self.head[3](self.head[2](torch.cat([self.head[1](self.head[0](x5)), x4], 1)))], 1)
        x_p4 = self.head[7](x_p4)  # C3 -> P4 output
        
        x_p5 = self.head[8](x_p4)  # Downsample
        x_p5 = torch.cat([x_p5, self.head[0](x5)], 1)
        x_p5 = self.head[9](x_p5)  # C3 -> P5 output
        
        # Detection
        return self.detect([x_p3, x_p4, x_p5])


def yolov7_base(nc=20, pretrained=False):
    """YOLOv7 base model (teacher)."""
    model = YOLOv7(nc=nc, width_multiple=1.0, depth_multiple=1.0)
    model.detect.stride = torch.tensor([8., 16., 32.])
    return model


def yolov7_small(nc=20, pretrained=False):
    """YOLOv7 small model (student)."""
    model = YOLOv7(nc=nc, width_multiple=0.5, depth_multiple=0.33)
    model.detect.stride = torch.tensor([8., 16., 32.])
    return model


def yolov7_tiny(nc=20, pretrained=False):
    """YOLOv7 tiny model (for aggressive pruning)."""
    model = YOLOv7(nc=nc, width_multiple=0.25, depth_multiple=0.33)
    model.detect.stride = torch.tensor([8., 16., 32.])
    return model


if __name__ == '__main__':
    # Test model creation
    print("Testing YOLOv7 model architectures...")
    
    # Test base model
    model_base = yolov7_base(nc=20)
    x = torch.randn(1, 3, 640, 640)
    
    model_base.eval()
    with torch.no_grad():
        y = model_base(x)
    
    print(f"\n✅ YOLOv7-Base:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output detections: {y[0].shape if isinstance(y, tuple) else y.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model_base.parameters())
    trainable_params = sum(p.numel() for p in model_base.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Test small model
    model_small = yolov7_small(nc=20)
    model_small.eval()
    with torch.no_grad():
        y = model_small(x)
    
    print(f"\n✅ YOLOv7-Small:")
    print(f"   Input shape: {x.shape}")
    print(f"   Output detections: {y[0].shape if isinstance(y, tuple) else y.shape}")
    
    total_params = sum(p.numel() for p in model_small.parameters())
    trainable_params = sum(p.numel() for p in model_small.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Model architecture tests passed!")
