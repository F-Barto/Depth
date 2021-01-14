from functools import partial

import torch.nn as nn

from networks.common.basic_blocks import PaddedConv3x3, disp_to_depth
from networks.predictor.base import MultiScaleBasePredictor


class InvDepthPredictor(nn.Module):
    def __init__(self, in_chans, prefix='', postfix=''):
        super(InvDepthPredictor, self).__init__()
        self.invdepthconv = PaddedConv3x3(in_chans, 1)
        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix

    def forward(self, x):

        output = self.sigmoid(self.invdepthconv(x))
        return {self.prefix+'inv_depth'+self.postfix: output}


class MultiScaleInvDepthPredictor(MultiScaleBasePredictor):
    def __init__(self, scales, in_chans, prefix='', postfix='', min_depth=0.1, max_depth=120.0):
        super(MultiScaleInvDepthPredictor, self).__init__(scales)

        assert len(in_chans) == scales

        self.sigmoid = nn.Sigmoid()

        self.prefix = prefix
        self.postfix = postfix

        self.invdepthconvs = nn.ModuleDict([
            (f'invdepthconv_{i}', PaddedConv3x3(in_chans[i], 1)) for i in range(scales)
        ])

        self.outputs = {i:None for i in range(scales)}

        self.scale_inv_depth = partial(disp_to_depth, min_depth=min_depth, max_depth=max_depth)

    def forward(self, x, i):
        if i >= self.scales: raise IndexError(f'The network has at most {self.scales} of prediction.')

        output = self.sigmoid(self.invdepthconvs[f'invdepthconv_{i}'](x))

        self.outputs[i] = output

    def get_prediction(self, i):
        if self.outputs[i] is  None:
            raise ValueError(f'Prediction of scale {i} not yet computed')

        return self.scale_inv_depth(self.outputs[i])[0]

    def compile_predictions(self):
        assert sum([v is not None for v in self.outputs.values()]) == self.scales, 'Not all scales have a prediction'

        values = list(self.outputs.values())
        scaled_values = [self.scale_inv_depth(v)[0] for v in values]

        if self.training:
            return {self.prefix+f'inv_depths'+self.postfix: scaled_values}
        else:
            return {self.prefix+f'inv_depths'+self.postfix: self.get_prediction(0)}