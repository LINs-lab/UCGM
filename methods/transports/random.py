import torch


class Random:

    def alpha_in(self, t):
        return torch.sin(t * 1.57)

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1 - torch.exp(-5 * t)
