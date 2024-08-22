import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, dilation):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = SeparableConv2d(in_channels, out_channels, 3, padding=6, dilation=6)
        self.conv3 = SeparableConv2d(in_channels, out_channels, 3, padding=12, dilation=12)
        self.conv4 = SeparableConv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv6 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = self.conv5(x5)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv6(x)
        return self.dropout(self.relu(self.bn(x)))

class DeepLabV3Plus(nn.Module):
    def __init__(self, num_classes=11, backbone='resnet50', weights = ResNet50_Weights.DEFAULT):
        super(DeepLabV3Plus, self).__init__()
        
        if backbone == 'resnet50':
            self.backbone = models.resnet50(weights = ResNet50_Weights.DEFAULT)
            low_level_inplanes = 256
            high_level_inplanes = 2048
        
        self.low_level_features = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
        )
        
        self.high_level_features = nn.Sequential(
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4,
        )
        
        self.aspp = ASPP(high_level_inplanes, 256)
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + low_level_inplanes, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1, bias=False)
        )

    def forward(self, x):
        size = x.size()[2:]
        
        low_level_feat = self.low_level_features(x)
        high_level_feat = self.high_level_features(low_level_feat)
        high_level_feat = self.aspp(high_level_feat)
        
        low_level_feat = F.interpolate(low_level_feat, size=high_level_feat.size()[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([high_level_feat, low_level_feat], dim=1)
        x = self.decoder(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)

