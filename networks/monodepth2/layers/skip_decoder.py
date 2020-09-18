# code from https://github.com/TRI-ML/packnet-sfm/blob/master/packnet_sfm/networks/layers/resnet/depth_decoder.py


from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn

from collections import OrderedDict
from .common import ConvBlock, Conv3x3, nearest_upsample, SubPixelUpsamplingBlock


class Conv1x1Block(nn.Module):
    """Layer to perform a convolution followed by activation function
    """
    def __init__(self, in_channels, out_channels, activation):
        super(Conv1x1Block, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=True)
        self.activation = activation(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.activation(out)
        return out

class SkipDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=range(4), num_output_channels=1, upsample_path='direct',
                 upsample_mode='nearest', blur=True, blur_at_end=True):
        super(SkipDecoder, self).__init__()

        self.num_output_channels = num_output_channels
        self.scales = scales

        available_upmodes = ['nearest', 'pixelshuffle', 'res-pixelshuffle']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        available_uppaths = ['direct', 'cascaded', 'conv1cascaded']
        if upsample_path not in available_uppaths:
            raise ValueError(f"upsample_path must be in {available_uppaths} | upsample_path={upsample_path}")
        self.upsample_path = upsample_path

        self.num_ch_enc = num_ch_enc
        self.num_ch_concat = sum(num_ch_enc[-4:])

        self.upscale_factors = np.zeros(len(num_ch_enc)).astype(int)
        if self.upsample_path == 'direct':
            self.upscale_factors[[-1,-2,-3]] = [8,4,2]
        else:
            self.upscale_factors[[-1, -2, -3]] = 2
        print(f"SkipDecoder upscale_factors={self.upscale_factors}")


        # decoder
        self.convs = OrderedDict()
        for i in range(4, 0, -1):
            # upconv_0, pre upsampling
            num_ch_in = self.num_ch_enc[i]

            if self.upsample_path == 'conv1cascaded':
                self.convs[("skipconv", i)] = Conv1x1Block(num_ch_in, num_ch_in, activation)

            if 'pixelshuffle' in self.upsample_mode:
                do_blur = blur and (i != 4 or blur_at_end)
                self.convs[("pixelshuffle", i)] = SubPixelUpsamplingBlock(num_ch_in, blur=do_blur,
                                                                          upscale_factor=self.upscale_factors[i])

        self.concatconv = ConvBlock(self.num_ch_concat, 256, activation)

        self.dispconv = Conv3x3(256, self.num_output_channels)

        if 'pixelshuffle' in self.upsample_mode:
            self.last_pixelshuffle = SubPixelUpsamplingBlock(self.num_output_channels, blur=do_blur, upscale_factor=4)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def upsample(self, x, i):

        if self.upsample_mode == 'pixelshuffle':
            x = self.convs[("pixelshuffle", i)](x)
        if self.upsample_mode == 'res-pixelshuffle':
            x = self.convs[("pixelshuffle", i)](x) + nearest_upsample(x, scale_factor=self.upscale_factors[i])
        if self.upsample_mode == 'nearest':
            x = nearest_upsample(x, scale_factor=self.upscale_factors[i])

        return x


    def forward(self, input_features):

        concat = self.upsample(input_features[-1], -1)

        for i in range(3, 0, -1):

            if self.upsample_path == 'conv1cascaded':
                skip = self.convs[("skipconv", i)](input_features[i])
            else:
                skip = input_features[i]

            if self.upsample_path == 'direct':
                if i > 1:
                    skip = self.upsample(skip, i)
                concat = [concat, skip]
                concat = torch.cat(concat, 1)
            else: # cascaded
                concat = [concat, skip]
                concat = torch.cat(concat, 1)
                if i > 1:
                    concat = self.upsample(concat, i)

        x = self.concatconv(concat)

        if self.upsample_mode == 'pixelshuffle':
            up_x = self.last_pixelshuffle(x)
        else:
            out_dispconv = self.dispconv(x)

        if self.upsample_mode == 'res-pixelshuffle':
            up_x = self.last_pixelshuffle(x) + nearest_upsample(out_dispconv, scale_factor=self.upscale_factors[i])
        if self.upsample_mode == 'nearest':
            up_x = nearest_upsample(out_dispconv, scale_factor=self.upscale_factors[i])

        self.disp = self.sigmoid(up_x)

        return self.disp