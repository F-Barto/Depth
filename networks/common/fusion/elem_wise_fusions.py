from networks.common.fusion.channels_equalizer import ModalitiesEqualizer
from networks.common.fusion.fusion_base import FusionBase


class ElemWiseMultFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, **kwargs):
        super().__init__(**kwargs)

        if activation_cls is not None:
            self.activation = activation_cls(inplace=True)
            if self.lidar_in_chans is not None and self.image_in_chans is not None:
                self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)
        else:
            self.activation = None
            self.equalizer = None

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls):
        self.activation = activation_cls(inplace=True)
        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, lidar_features):

        image_features, lidar_features = self.equalizer(image_features, lidar_features)

        return image_features * lidar_features



class ElemWiseSumFusion(FusionBase):

    def __init__(self, lidar_in_chans=None, image_in_chans=None, activation_cls=None, **kwargs):
        super().__init__(**kwargs)

        if activation_cls is not None:
            self.activation = activation_cls(inplace=True)
            if self.lidar_in_chans is not None and self.image_in_chans is not None:
                self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)
        else:
            self.activation = None
            self.equalizer = None

    def setup_module(self, lidar_in_chans, image_in_chans, activation_cls):
        self.activation = activation_cls(inplace=True)
        self.equalizer = ModalitiesEqualizer(lidar_in_chans, image_in_chans, activation_cls)

    @property
    def require_chans(self):
        return True

    @property
    def require_activation(self):
        return True

    def forward(self, image_features, lidar_features):

        image_features, lidar_features = self.equalizer(image_features, lidar_features)

        return image_features, lidar_features


