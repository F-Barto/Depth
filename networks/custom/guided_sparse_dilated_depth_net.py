import torch.nn as nn

from functools import partial

from networks.custom.layers.dilated_resnet import resnet18
from networks.custom.layers.sparse_conv_encoder import SparseConvEncoder, SparseConv1x1
from networks.custom.layers.skip_decoder import SkipDecoder
from networks.monodepth2.layers.common import disp_to_depth, get_activation
from networks.attention_guidance import AttentionGuidance

from utils.depth import depth2inv

########################################################################################################################


class GuidedSparseDepthResNet(nn.Module):
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
    def __init__(self, input_channels=3, activation='relu', guidance='attention', attention_scheme='res-sig',
                 inverse_lidar_input=True, dilation_rates=None, combination='sum', **kwargs):
        super().__init__()

        assert guidance in ['attention', 'continuous']

        self.inverse_lidar_input = inverse_lidar_input

        activation_cls = get_activation(activation)

        # keeping the name `encoder` so that we can use pre-trained weight directly
        self.encoder = resnet18(activation_cls, input_channels=input_channels)
        self.lidar_encoder = SparseConvEncoder([3,3,3,3], activation_cls,
                                               dilation_rates=dilation_rates, combination=combination)

        self.num_ch_enc = self.encoder.num_ch_enc

        self.extend_lidar = nn.ModuleDict()
        for i in range(len(self.num_ch_enc )):
            in_chans = self.lidar_encoder.num_ch_enc[i]
            out_chans = self.num_ch_enc[i]

            self.extend_lidar.update({
                f"extend_lidar{i}": SparseConv1x1(in_chans, out_chans ,activation_cls)
            })

        skip_features_factor = 2 if ('concat' in attention_scheme) else 1
        self.num_ch_skips = [skip_features_factor * num_ch for num_ch in self.num_ch_enc]

        # at each resblock fuse with guidance the features of both encoders
        self.guidances = nn.ModuleDict()
        for i in range(len(self.num_ch_enc)):

            num_ch =  self.num_ch_enc[i]

            if guidance == 'attention':
                self.guidances.update({f"guidance_{i}": AttentionGuidance(num_ch, activation_cls, attention_scheme)})
            else:
                print(f"guidance {guidance} not implemented")

        self.decoder = SkipDecoder(num_ch_enc=self.num_ch_skips, activation=activation_cls, **kwargs)

        self.scale_inv_depth = partial(disp_to_depth, min_depth=0.1, max_depth=120.0)

    def forward(self, cam_input, lidar_input):
        """
        Runs the network and returns inverse depth maps
        """

        nb_features = len(self.num_ch_enc)

        if self.inverse_lidar_input:
            lidar_input = depth2inv(lidar_input)

        cam_features = self.encoder(cam_input)
        lidar_features = self.lidar_encoder(lidar_input)

        extended_lidar_features = [self.extend_lidar[f"extend_lidar{i}"](lidar_features[i])[0]
                                   for i in range(nb_features)]
        lidar_features = extended_lidar_features

        self.guided_features = []
        for i in range(nb_features):
            guided_feature = self.guidances[f"guidance_{i}"](cam_features[i], lidar_features[i])
            self.guided_features.append(guided_feature)

        outputs = self.decoder(self.guided_features)

        outputs = {k: self.scale_inv_depth(v)[0] for k,v in outputs.items()}

        return outputs