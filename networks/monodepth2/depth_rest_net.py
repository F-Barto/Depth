import torch.nn as nn
from functools import partial

from networks.monodepth2.layers.resnet_encoder import ResnetEncoder
from networks.monodepth2.layers.depth_decoder import DepthDecoder
from networks.monodepth2.layers.common import disp_to_depth, get_activation


########################################################################################################################

class DepthResNet(nn.Module):
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
    def __init__(self, num_layers=18, activation='relu'):
        super().__init__()

        assert num_layers in [18, 34, 50], 'ResNet version {} not available'.format(num_layers)

        activation_cls = get_activation(activation)

        self.encoder = ResnetEncoder(num_layers=num_layers, activation=activation_cls)
        self.decoder = DepthDecoder(num_ch_enc=self.encoder.num_ch_enc, activation=activation_cls)
        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=120.0)

    def forward(self, x):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x = self.encoder(x)
        x = self.decoder(x)
        disps = [x[('disp', i)] for i in range(4)]

        if self.training:
            return [self.scale_inv_depth(d)[0] for d in disps]
        else:
            return self.scale_inv_depth(disps[0])[0]