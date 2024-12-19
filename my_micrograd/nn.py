import numpy as np
from my_micrograd.engine import Value

class Neuron:

    def __init__(self, n_inputs):
        self.w = [Value(np.random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(np.random.uniform(-1, 1))

    def __call__(self, x):
        # w * x + b
        assert len(x) == len(self.w), f"Inputs 'x' must have be of length {len(self.w)}."
        y = sum((x[i] * self.w[i] for i in range(len(self.w))), self.b)
        return y.tanh()

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, n_inputs, n_outputs):
        self.n_inputs = n_inputs
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        assert len(x) == self.n_inputs, f"Inputs 'x' must be of length {self.n_inputs}."
        y = [neuron(x) for neuron in self.neurons]
        return y

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:

    def __init__(self, n_inputs, n_outputs): # n_outputs: list of number of neurons per layer
        self.layers = []
        self.n_inputs = n_inputs
        prev_layer_n_outputs = n_inputs
        for num_neurons in n_outputs:
            self.layers.append(Layer(prev_layer_n_outputs, num_neurons))
            prev_layer_n_outputs = num_neurons

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
