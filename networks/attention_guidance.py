from torch import nn
import torch


class AttentionBlock(nn.Module):
    def __init__(self, inplanes, activation_cls, attention_scheme='res-sig',):
        super().__init__()

        self.activation = activation_cls(inplace=True)

        if 'sig' in attention_scheme:
            self.act = nn.Sigmoid()
        elif 'tan' in attention_scheme:
            self.act = nn.Tanh()
        else:
            raise ValueError(f'Last activation choice invalid either sig or tanh: {attention_scheme}')

        if 'concat' in attention_scheme:
            planes = inplanes
        else:
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
    def __init__(self, inplanes, activation_cls, attention_scheme='res-sig'):
        super().__init__()

        if 'concat' in self.attention_scheme:
            self.attention_block = AttentionBlock(inplanes * 2, activation_cls, attention_scheme)
        else:
            self.lidar_attention_block = AttentionBlock(inplanes * 2, activation_cls, attention_scheme)
            self.image_attention_block = AttentionBlock(inplanes * 2, activation_cls, attention_scheme)

        self.attention_scheme = attention_scheme

    def fuse_features(self, original_features, attentive_masks):
        if 'res' in self.attention_scheme:
            residual_features = [of * am + of for of,am in zip(original_features, attentive_masks)]
            return sum(residual_features)
        elif 'mult' in self.attention_scheme:
            features = [of * am for of, am in zip(original_features, attentive_masks)]
            return sum(features)
        else:
            raise ValueError(f'Attention scheme invalid either res or mult: {self.attention_scheme}')

    def forward(self, image_features, lidar_features):

        c = torch.cat([image_features, lidar_features], dim=1)

        if 'concat' in self.attention_scheme:
            attentive_mask = self.attention_block(c)
            final_features = self.fuse_features([c], [attentive_mask])
        else:
            image_attentive_mask = self.image_attention_block(c)
            lidar_attentive_mask = self.lidar_attention_block(c)
            attentive_masks = [image_attentive_mask, lidar_attentive_mask]
            final_features = self.fuse_features([image_features, lidar_features], attentive_masks)

        return final_features

