# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py

# Adapted from monodepth2
# https://github.com/nianticlabs/monodepth2/blob/master/networks/depth_decoder.py

from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .common import ConvBlock, Conv3x3, nearest_upsample, SubPixelUpsamplingBlock


class DepthDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=range(4), num_output_channels=1, use_skips=True,
                 concat_skips=True, two_encoder=False, upsample_mode='nearest', blur=True, blur_at_end=True):
        super(DepthDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.concat_skips = concat_skips
        self.scales = scales

        if upsample_mode not in ['nearest', 'pixelshuffle']:
            raise ValueError(f"upsample_mode must be in ['nearest', 'pixelshuffle'] | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode



        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16,32,64,128,256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0, pre upsampling
            if i == 4:
                num_ch_in = self.num_ch_enc[-1]
                if two_encoder and concat_skips:
                    num_ch_in += self.num_ch_enc[-1]
            else:
                num_ch_in = self.num_ch_dec[i + 1]

            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out, activation)

            if self.upsample_mode == 'pixelshuffle':
                do_blur = blur and (i != 4 or blur_at_end)
                self.convs[("pixelshuffle", i)] = SubPixelUpsamplingBlock(num_ch_out, upscale_factor=2, blur=do_blur)

            # upconv_1, post upsampling
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0 and self.concat_skips:
                num_ch_in += self.num_ch_enc[i - 1]
                if two_encoder:
                    num_ch_in += self.num_ch_enc[i - 1]

            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out, activation)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(4, -1, -1):

            print()
            print(f"input_features[- 1]: {input_features[i - 1].shape}")
            print(f"x {i}: {x.shape}")
            print()

            x = self.convs[("upconv", i, 0)](x)

            if self.upsample_mode == 'pixelshuffle':
                x = self.convs[("pixelshuffle", i)](x)
            if self.upsample_mode == 'nearest':
                x = nearest_upsample(x)

            if self.use_skips and i > 0:
                if self.concat_skips:
                    x = [x, input_features[i - 1]]
                    x = torch.cat(x, 1)
                else:
                    x = x + input_features[i - 1]

            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs[("disp", i)] = self.sigmoid(self.convs[("dispconv", i)](x))

        return self.outputs