from networks.common.fusion.channels_equalizer import ModalitiesEqualizer

import torch.nn as nn


class ElemWiseMultFusion(nn.Module):

    def __init__(self, lidar_in_chans, image_in_chans, activation_cls, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation_cls(inplace=True)

        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    def forward(self, image_features, lidar_features):

        image_features, lidar_features = self.equalizer(image_features, lidar_features)

        return image_features * lidar_features


class ElemWiseSumFusion(nn.Module):

    def __init__(self, lidar_in_chans, image_in_chans, activation_cls, **kwargs):
        super().__init__(**kwargs)

        self.activation = activation_cls(inplace=True)

        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    def forward(self, image_features, lidar_features):

        image_features, lidar_features = self.equalizer(image_features, lidar_features)

        return image_features, lidar_features


