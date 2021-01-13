import numpy as np
import torch
import torch.nn as nn

from networks.common.basic_blocks import PaddedConv3x3Block, nearest_upsample, SubPixelUpsamplingBlock
from networks.predictor.common import create_multiscale_predictor


class MultiscalePredictionDecoder(nn.Module):
    def __init__(self, num_ch_enc, activation, scales=4, predictor='inv_depth', upsample_mode='nearest',
                 blur=True, blur_at_end=True):
        super(MultiscalePredictionDecoder, self).__init__()

        self.scales = scales

        available_upmodes = ['nearest', 'pixelshuffle', 'res-pixelshuffle']
        if upsample_mode not in available_upmodes:
            raise ValueError(f"upsample_mode must be in {available_upmodes} | upsample_mode={upsample_mode}")
        self.upsample_mode = upsample_mode

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16,32,64,128,256])

        # decoder
        self.convs = nn.ModuleDict()
        for i in range(self.scales, -1, -1):
            # upconv_0, pre upsampling
            num_ch_in = self.num_ch_enc[-1] if i == self.scales else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_0"] = PaddedConv3x3Block(num_ch_in, num_ch_out, activation)

            if 'pixelshuffle' in self.upsample_mode:
                do_blur = blur and (i != 0 or blur_at_end)
                self.convs[f"pixelshuffle_i"] = SubPixelUpsamplingBlock(num_ch_out, upscale_factor=2, blur=do_blur)

            # upconv_1, post upsampling
            num_ch_in = self.num_ch_dec[i] + self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[f"upconv_{i}_1"] = PaddedConv3x3Block(num_ch_in, num_ch_out, activation)

        self.predictor = create_multiscale_predictor(predictor, self.scales, in_chans=self.num_ch_dec[:self.scales])

    def forward(self, input_features):
        self.outputs = {}

        # decoder
        x = input_features[-1]
        for i in range(self.scales, -1, -1):

            if self.refine_preds and i < 3:
                x = [x, self.outputs[("disp", i+1)]]
                if self.uncertainty:
                    x.append(self.outputs[("uncertainty", i+1)])
                x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 0)](x)

            if self.upsample_mode == 'pixelshuffle':
                x = self.convs[("pixelshuffle", i)](x)
            if self.upsample_mode == 'res-pixelshuffle':
                x = self.convs[("pixelshuffle", i)](x) + nearest_upsample(x)
            if self.upsample_mode == 'nearest':
                x = nearest_upsample(x)

            x = [x, input_features[i - 1]]
            x = torch.cat(x, 1)

            x = self.convs[("upconv", i, 1)](x)

            if i in range(self.scales):
                self.predictor(x, i)

        return self.predictor.compile_predictions()