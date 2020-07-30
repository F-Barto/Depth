import torch.nn as nn

from functools import partial

from networks.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.monodepth2.layers.depth_decoder import DepthDecoder
from networks.monodepth2.layers.common import disp_to_depth, get_activation
from networks.pac import PacConv2d
from networks.attention_guidance import AttentionGuidance

from utils.depth import depth2inv

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
    def __init__(self, num_layers=18, input_channels=3, activation='relu', guidance='pac', attention_scheme='res-sig',
                 inverse_lidar_input=True, preact=False, invertible=False, n_power_iterations=5, **kwargs):
        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        assert guidance in ['pac', 'attention']

        self.inverse_lidar_input = inverse_lidar_input

        activation_cls = get_activation(activation)

        # keeping the name `encoder` so that we can use pre-trained weight directly
        self.encoder = ResnetEncoder(num_layers=num_layers, input_channels=input_channels, activation=activation_cls,
                                     preact=preact, invertible=invertible, n_power_iterations=n_power_iterations)
        self.lidar_encoder = ResnetEncoder(num_layers=num_layers, input_channels=1, activation=activation_cls,
                                           no_first_norm=True, preact=preact, invertible=invertible,
                                           n_power_iterations=n_power_iterations)

        self.num_ch_enc = self.encoder.num_ch_enc
        skip_features_factor = 2 if 'concat' in attention_scheme else 1
        self.num_ch_skips = [skip_features_factor * num_ch for num_ch in self.num_ch_enc]

        # at each resblock fuse with guidance the features of both encoders
        self.guidances = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

            num_ch =  self.num_ch_enc[i]

            if guidance == 'pac':
                self.guidances.update({f"guidance_{i}" : PacConv2d(num_ch, num_ch, 3, padding=1, native_impl=False)})
            elif guidance == 'attention':
                self.guidances.update({f"guidance_{i}": AttentionGuidance(num_ch, activation_cls, attention_scheme)})
            else:
                print(f"guidance {guidance} not implemented")

        self.decoder = DepthDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=120.0)

    def forward(self, cam_input, lidar_input):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """

        if self.inverse_lidar_input:
            lidar_input = depth2inv(lidar_input)

        cam_features = self.encoder(cam_input)
        lidar_features = self.lidar_encoder(lidar_input)

        self.guided_features = []
        for i in range(len(self.num_ch_enc)):
            guided_feature = self.guidances[f"guidance_{i}"](cam_features[i], lidar_features[i])
            self.guided_features.append(guided_feature)

        x = self.decoder(self.guided_features)

        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]