import torch
import torch.nn as nn
from scipy.special import beta


class PoincareGLU(nn.Module):
    def __init__(self, ball=None):
        super().__init__()
        self.ball = ball
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, x, dim=-1):
        channels = x.size(dim)
        beta_n = beta(channels / 2, 1 / 2)
        beta_ni = beta(channels / 4, 1 / 2)
        xa, xb = (self.ball.logmap0(x, dim=dim) * beta_ni / beta_n).chunk(2, dim=dim)
        return self.ball.expmap0(xa * (xb * (channels ** 0.5) * self.scale.exp()).sigmoid(), dim=dim)