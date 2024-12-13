import numpy as np
from abc import ABC, abstractmethod
from engine import Value

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

class LossFunction(ABC):
    """ Loss Function Interface """
    @abstractmethod
    def __call__(y_pred, y)-> Value:
        pass


class MeanSquaredError(LossFunction):
    """ Mean Squared Error implementation of LossFunction """
    @staticmethod
    def __call__(y_pred, y):
        return sum((y_out - y_true)**2 for y_out, y_true in zip(y_pred, y))


class Optimizer:
    def __init__(self, model:MLP, loss_fct:LossFunction=lambda y_pred, y: sum((y_out - y_true)**2 for y_out, y_true in zip(y_pred, y))):
        self.model = model
        self.loss_fct = loss_fct

    def _step(self, X, y, learning_rate):
        # Forward pass
        y_pred = [self.model(x) for x in X]
        loss = self.loss_fct(y_pred, y)

        # Backward pass
        for p in self.model.parameters():
            p.grad = 0
        loss.backward()
        for p in self.model.parameters():
            p.data -= learning_rate * p.grad
        return loss

    def fit(self, X, y, epochs, learning_rate=1e-3):
        # Parameter check
        assert all(len(x) == self.model.layers[0].n_inputs for x in X), "Input X has wrong shape"
        assert all(isinstance(y_i, Value) for y_i in y), "Each target y_i must be a Value"

        history = []
        for _ in range(epochs):
            history.append(self._step(X, y, learning_rate))
        return history