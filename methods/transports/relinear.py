class ReLinear:

    def alpha_in(self, t):
        return 1 - t

    def gamma_in(self, t):
        return t

    def alpha_to(self, t):
        return -1

    def gamma_to(self, t):
        return 1
