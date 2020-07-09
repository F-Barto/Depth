import torch
import torch.nn as nn

from functools import partial
from collections import OrderedDict

from networks.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.monodepth2.layers.depth_decoder import DepthDecoder
from networks.monodepth2.layers.common import disp_to_depth, get_activation
from networks.pac import PacConv2d
from networks.attention_guidance import AttentionGuidance

########################################################################################################################


class GuidedDepthResNet(nn.Module):
    """
    Inverse depth network based on the ResNet architecture.
    Parameters
    ----------
    version : str
        Has a XY format, where:
        X is the number of residual layers [18, 34, 50] and
        Y is an optional ImageNet pretrained flag added by the "pt" suffix
        Example: "18pt" initializes a pretrained ResNet18, and "34" initializes a ResNet34 from scratch
    kwargs : dict
        Extra parameters
    """
    def __init__(self, num_layers=18, input_channels=3, activation='relu', guidance='pac'):
        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        assert guidance in ['pac', 'attention']

        activation_cls = get_activation(activation)

        # keeping the name `encoder` so that we can use pre-trained weight directly
        self.encoder = ResnetEncoder(num_layers=num_layers, input_channels=input_channels,
                                     activation=activation_cls)
        self.lidar_encoder = ResnetEncoder(num_layers=num_layers, input_channels=1,
                                           activation=activation_cls, no_first_norm=True)

        self.num_ch_enc = self.encoder.num_ch_enc

        # at each resblock fuse with guidance the features of both encoders
        self.pacs = OrderedDict()
        for i in range(len(self.num_ch_enc)):
            # upconv_0
            num_ch =  self.num_ch_enc[i]

            if guidance == 'pac':
                self.guidances[("guidance", i)] = PacConv2d(num_ch, num_ch, 3, padding=1, native_impl=False)
            elif guidance == 'attentioin':
                self.guidances[("guidance", i)] = AttentionGuidance(num_ch, activation_cls)
            else:
                print(f"guidance {guidance} not implemented")

        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, activation=activation_cls)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=100.0)

    def forward(self, cam_input, lidar_input):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        cam_features = self.encoder(cam_input)
        lidar_features = self.lidar_encoder(lidar_input)

        self.guided_features = []
        for i in range(len(self.num_ch_enc)):
            guided_feature = self.guidances[("guidance", i)](cam_features[i], lidar_features[i])
            self.guided_features.append(guided_feature)

        x = self.decoder(self.guided_features)

        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]