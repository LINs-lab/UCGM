class EDM:

    def alpha_in(self, t):
        sigma = ((t * 2.68 - 1.59) * 4).exp()
        return sigma / (sigma**2 + 0.25) ** (1 / 2)

    def gamma_in(self, t):
        sigma = ((t * 2.68 - 1.59) * 4).exp()
        return 1 / (sigma**2 + 0.25) ** (1 / 2)

    def alpha_to(self, t):
        sigma = ((t * 2.68 - 1.59) * 4).exp()
        return -0.5 / (sigma**2 + 0.25) ** (1 / 2)

    def gamma_to(self, t):
        sigma = ((t * 2.68 - 1.59) * 4).exp()
        return 2 * sigma / (sigma**2 + 0.25) ** (1 / 2)
