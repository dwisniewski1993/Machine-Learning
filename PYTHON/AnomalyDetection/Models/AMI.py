from numpy import array


class AbstractModelInterface:
    def train(self):
        pass

    def score(self, data: array):
        pass
