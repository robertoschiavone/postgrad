import random

from value import Value


class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, x):
        return (sum((wi * xi for wi, xi in zip(self.w, x)), self.b)).tanh()

    def parameters(self):
        return self.w + [self.b]
