import numpy as np
from my_micrograd.nn import MLP
from my_micrograd.loss import mean_squared_error
from abc import ABC, abstractmethod
from collections.abc import Iterable
from my_micrograd.utils import get_list_dimensions

class Optimizer(ABC):

    def __init__(self, model: MLP, loss_fct=mean_squared_error):
        self.model = model
        self.loss_fct = loss_fct

    @abstractmethod
    def fit(self, X, y, epochs, *args, **kwargs):
        pass

    @staticmethod
    def _shuffle(X, y):
        if X is None or y is None:
            raise ValueError("X and y cannot be None")
        if not len(X) == len(y):
            raise ValueError("X and y must have same length")
        indices = np.arange(len(X))
        np.random.shuffle(indices)

        X_ = X.copy()
        y_ = y.copy()
        for idx, i in enumerate(indices):
            X_[i], y_[i] = X[idx], y[idx]
        return X_, y_

class BatchGradientDescent(Optimizer):

    def _step(self, X, y, learning_rate):
        # Set gradients to zero
        for p in self.model.parameters():
            p.zero_grad()

        # Forward pass
        y_pred = [self.model(x)[0] for x in X]
        loss = self.loss_fct(y_pred, y)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            p.data -= learning_rate * p.grad

        return loss

    def fit(self, X, y, epochs, learning_rate=1e-3):
        # Parameter check
        if not all(len(x) == self.model.layers[0].n_inputs for x in X):
            raise ValueError("Input X has wrong shape")
        if len(y) != len(X):
            raise ValueError("X and y must have same length")

        history = []
        for _ in range(epochs):
            history.append(self._step(X, y, learning_rate))

        return history


class StochasticGradientDescent(Optimizer):

    def __init__(self, model:MLP, loss_fct=mean_squared_error, momentum=0.0):
        super().__init__(model, loss_fct)
        self.momentum = momentum
        self._velocity = {p: 0.0 for p in self.model.parameters()}

    def _step(self, x, y, learning_rate):
        # Set gradients to zero
        for p in self.model.parameters():
            p.zero_grad()

        # Forward pass
        y_pred = self.model(x)[0]
        loss = self.loss_fct(y_pred, y)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            self._velocity[p] = self.momentum * self._velocity[p] - learning_rate * p.grad
            p.data += self._velocity[p]

        return loss

    def fit(self, X, y, epochs, learning_rate=1e-3):
        # Parameter check
        assert all(len(x) == self.model.layers[0].n_inputs for x in X), "Input X has wrong shape"
        # assert all(isinstance(y_i, Value) for y_i in y), "Each target y_i must be a Value"

        history = []
        for _ in range(epochs):
            epoch_loss = 0.0
            for x, y_ in zip(X, y):
                epoch_loss += self._step(x, y_, learning_rate)
            history.append(epoch_loss)

        return history


class MiniBatchGradientDescent(Optimizer):

    def _step(self, X, y, learning_rate):
        # Set gradients to zero
        for p in self.model.parameters():
            p.zero_grad()

        # Forward pass
        input_dim = get_list_dimensions(X)
        model_input_len = self.model.n_inputs
        if input_dim == 1:
            # Stochastic approach
            if model_input_len != len(X):
                raise ValueError(f"X must have same length as the models input layer: {model_input_len}")
            y_pred = self.model(X)[0]
        elif input_dim >= 2:
            # (mini)batch approach
            if not all(len(input) == model_input_len for input in X):
                raise ValueError(f"Inputs must be of same length as the models input layer: {model_input_len}")
            y_pred = [self.model(x)[0] for x in X]

        loss = self.loss_fct(y_pred, y)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            p.data -= learning_rate * p.grad

        return loss

    def fit(self, X, y, epochs, learning_rate=1e-3, batch_size=None):
        # Parameter check
        if not all(len(x) == self.model.layers[0].n_inputs for x in X):
            raise ValueError("Input X has wrong shape")
        if len(y) != len(X):
            raise ValueError("X and y must have same length")
        if batch_size is not None and batch_size > len(X):
            raise ValueError("Batch size cannot be greater than number of samples")

        history = []
        # loop for batch_size=None -> batch gradient descent
        if batch_size is None:
            for _ in range(epochs):
                history.append(self._step(X, y, learning_rate))
        # Loop for batch_size=1 -> stochastic gradient descent
        elif batch_size == 1:
            for _ in range(epochs):
                epoch_loss = 0.0
                for x, y_ in zip(X, y):
                    epoch_loss += self._step(x, y_, learning_rate)
                history.append(epoch_loss)
        # Loop for batch_size>1 -> mini-batch gradient descent
        else:
            batches_per_epoch = int(len(X) / batch_size)
            for _ in range(epochs):
                X_shuffled, y_shuffled = self._shuffle(X, y)
                for i in range(batches_per_epoch):
                    X_ = X_shuffled[i * batch_size:(i + 1) * batch_size]
                    y_ = y_shuffled[i * batch_size:(i + 1) * batch_size]
                    history.append(self._step(X_, y_, learning_rate))
                if len(X) % batch_size != 0:
                    X_, y_ = X_shuffled[batches_per_epoch*batch_size:][:], y_shuffled[batches_per_epoch*batch_size:]
                    history.append(self._step(X_, y_, learning_rate))

        return history


class AdaGrad(MiniBatchGradientDescent):

    def __init__(self, model: MLP, loss_fct=mean_squared_error, epsylon=1e-5):
        super().__init__(model, loss_fct)
        self._squared_gradients = {p: 0.0 for p in self.model.parameters()}
        self.epsylon = epsylon # Term for numeric stability

    def _step(self, X, y, learning_rate):
        # Set gradients to zero
        for p in self.model.parameters():
            p.zero_grad()

        # Forward pass
        input_dim = get_list_dimensions(X)
        model_input_len = self.model.n_inputs
        if input_dim == 1:
            # Stochastic approach
            if model_input_len != len(X):
                raise ValueError(f"X must have same length as the models input layer: {model_input_len}")
            y_pred = self.model(X)[0]
        elif input_dim >= 2:
            # (mini)batch approach
            if not all(len(input) == model_input_len for input in X):
                raise ValueError(f"Inputs must be of same length as the models input layer: {model_input_len}")
            y_pred = [self.model(x)[0] for x in X]

        loss = self.loss_fct(y_pred, y)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            self._squared_gradients[p] += p.grad ** 2
            p.data -= (learning_rate / np.sqrt(self._squared_gradients[p] + self.epsylon)) * p.grad

        return loss


class RMSprop(MiniBatchGradientDescent):

    def __init__(self, model: MLP, loss_fct=mean_squared_error, decay_rate=0.9, epsylon=1e-5):
        super().__init__(model, loss_fct)
        self.decay_rate = decay_rate
        self.epsylon = epsylon  # Term for numeric stability
        self._squared_gradients = {p: 0.0 for p in self.model.parameters()}

    def _step(self, X, y, learning_rate):
        # Set gradients to zero
        for p in self.model.parameters():
            p.zero_grad()

        # Forward pass
        input_dim = get_list_dimensions(X)
        model_input_len = self.model.n_inputs
        if input_dim == 1:
            # Stochastic approach
            if model_input_len != len(X):
                raise ValueError(f"X must have same length as the models input layer: {model_input_len}")
            y_pred = self.model(X)[0]
        elif input_dim >= 2:
            # (mini)batch approach
            if not all(len(input) == model_input_len for input in X):
                raise ValueError(f"Inputs must be of same length as the models input layer: {model_input_len}")
            y_pred = [self.model(x)[0] for x in X]

        loss = self.loss_fct(y_pred, y)

        # Backward pass
        loss.backward()
        for p in self.model.parameters():
            self._squared_gradients[p] = self.decay_rate * self._squared_gradients[p] + (1 - self.decay_rate) * p.grad**2
            p.data -= (learning_rate / np.sqrt(self._squared_gradients[p] + self.epsylon)) * p.grad

        return loss


class Adam(Optimizer):
    pass