from torch import nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, inplanes, activation_cls, attention_scheme='res-sig'):
        super().__init__()

        self.activation = activation_cls(inplace=True)

        if 'sig' in attention_scheme:
            self.act = nn.Sigmoid()
        elif 'tan' in attention_scheme:
            self.act = nn.Tanh()
        else:
            raise ValueError(f'Last activation choice invalid either sig or tanh: {attention_scheme}')

        self.attention_scheme = attention_scheme

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

        out = self.act(x)

        return out

class AttentionGuidance(nn.Module):
    def __init__(self, inplanes, activation_cls):
        super().__init__()

        self.lidar_attention_block = AttentionBlock(inplanes * 2, activation_cls)
        self.image_attention_block = AttentionBlock(inplanes * 2, activation_cls)

    def forward(self, image_features, lidar_features):

        c = torch.cat([image_features, lidar_features], dim=1)

        if 'res' in self.attention_scheme:
            new_image_features = image_features * self.image_attention_block(c) + image_features
            new_lidar_features = lidar_features * self.lidar_attention_block(c) + lidar_features
        elif 'mult' in self.attention_scheme:
            new_image_features = image_features * self.image_attention_block(c)
            new_lidar_features = lidar_features * self.lidar_attention_block(c)
        else:
            raise ValueError(f'Attention scheme invalid either res or mult: {self.attention_scheme}')

        final_features = new_image_features + new_lidar_features

        return final_features

