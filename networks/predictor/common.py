import abc

from .invdepth import InvDepthPredictor, MultiScaleInvDepthPredictor

def create_multiscale_predictor(predictor_name, scales, **kwargs):
    assert scales > 1

    if predictor_name == 'inv_depth':
        return MultiScaleInvDepthPredictor(scales, **kwargs)


class MultiScaleBasePredictor(nn.Module):
    def __init__(self, scales):
        super(MultiScaleInvDepthPredictor, self).__init__()
        self.scales = scales

    @abc.abstractmethod
    def forward(self, x, i):
        pass

    @abc.abstractmethod
    def get_prediction(self, i):
        pass

    @abc.abstractmethod
    def compile_predictions(self):
        pass