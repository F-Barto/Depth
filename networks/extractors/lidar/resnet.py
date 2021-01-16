import numpy as np

from networks.common.resnet_base import ResNetBase
from networks.common.basic_blocks import conv7x7


class LiDARResNetExtractor(ResNetBase):

    """
    In this version we remove the first batch norm as the input is expected to be projected sparse LiDAR.
    As such, the majority of input values being 0s encoding 'invalid LiDAR pixels',
    the batch norm statistics have no senses.

    The 'small' argument indicates if we remove or not the last two ResGroup of resolution 1/16 and 1/32.
    """

    def __init__(self, block, layers, activation, zero_init_residual=False, input_channels=3, small=True,
                 **kwargs):
        super(ResNetBase, self).__init__(**kwargs)

        self.small = small
        self.num_ch_enc = np.array([64, 64, 128])

        if not self.small:
            self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        self.inplanes = 64

        ############### first conv ###############
        self.conv1 = conv7x7(input_channels, self.inplanes, stride=2, bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.activation = activation(inplace=True)

        ############### body ###############
        self.layer1 = self._make_layer(block, 64, layers[0], activation, stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], activation, stride=2)

        if not self.small:
            self.layer3 = self._make_layer(block, 256, layers[2], activation, stride=2)
            self.layer4 = self._make_layer(block, 512, layers[3], activation, stride=2)

        self.init_weights(zero_init_residual)

    def forward(self, x):

        self.features = []
        x = self.conv1(x)
        self.features.append(self.activation(x))

        self.features.append(self.layer1(self.features[-1]))
        self.features.append(self.layer2(self.features[-1]))
        if not self.small:
            self.features.append(self.layer3(self.features[-1]))
            self.features.append(self.layer4(self.features[-1]))

        return self.features