from torch import nn
import torch

class AttentionBlock(nn.Module):
    def __init__(self, inplanes, activation_cls):
        super().__init__()

        self.activation = activation_cls(inplace=True)
        self.sig = nn.Sigmoid()

        planes= inplanes//2

        self.conv_1x1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv_3x3 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

    def forward(self, x):
        x = self.conv_1x1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv_3x3(x)
        x = self.bn2(x)

        out = self.sig(x)

        return out

class AttentionGuidance(nn.Module):
    def __init__(self, inplanes, activation_cls):
        super().__init__()

        self.lidar_attention_block = AttentionBlock(inplanes * 2, activation_cls)
        self.image_attention_block = AttentionBlock(inplanes * 2, activation_cls)

    def forward(self, image_features, lidar_features):

        c = torch.cat([image_features, lidar_features], dim=1)

        new_image_features = image_features * self.image_attention_block(c) + image_features
        new_lidar_features = lidar_features * self.lidar_attention_block(c) + lidar_features

        final_features = new_image_features + new_lidar_features

        return final_features

