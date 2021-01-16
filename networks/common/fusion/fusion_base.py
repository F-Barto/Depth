import torch.nn as nn

class FusionBase(nn.Module):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def require_chans(self):
        raise NotImplementedError

    @property
    def require_activation(self):
        raise NotImplementedError