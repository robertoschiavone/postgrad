from neuron import Neuron


class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out if len(out) > 1 else out[0]

    def parameters(self):
        return [parameter for neuron in self.neurons
                for parameter in neuron.parameters()]
