import torch
import torch.nn as nn

class BaseRBM(nn.Module):
    def __init__(self, n_visible, n_hidden, k=1):
        super().__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.k = k
        self.W = nn.Parameter(torch.randn(n_hidden, n_visible) * 0.01)
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

    def v_to_h(self, v):
        raise NotImplementedError

    def h_to_v(self, h):
        raise NotImplementedError

    def forward(self, v):
        # Can implement contrastive divergence generically if desired
        raise NotImplementedError