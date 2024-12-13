from my_micrograd.engine import Value
import numpy as np

class LinearRegression:
    def __init__(self, num_features):
        self.num_features = num_features
        self.weights = [Value(np.random.uniform(-1, 1)) for _ in range(num_features)]
        # setting the labels
        for w in self.weights:
            w.label = "w"


    def predict(self, X):
        assert len(X[0]) == self.num_features, "wrong number of inputs"
        return [np.dot(x, self.weights) for x in X]


    def fit(self, X, y, epochs, learning_rate=0.001): # Batch Gradient Descent
        assert len(X[0]) == self.num_features, "wrong number of inputs"
        assert len(X) == len(y), "X and y must have same length"

        history = []
        for _ in range(epochs):
            # Forward pass
            loss = self._mean_squared_error(X, y)
            history.append(loss)

            # Backward pass
            self._reset_grad()
            loss.backward()
            for w in self.weights:
                w.data -= learning_rate * w.grad
        return history


    def _loss(self, x, y):
        assert x != None, "x cannot be None"
        assert y is not None, "y cannot be None"
        assert len(x[0]) == self.num_features, "wrong number of inputs"
        return self.predict(x) - y


    def _mean_squared_error(self, X, y): # C = 1/2M sum((h(x) - y)^2)
        assert len(X[0]) == self.num_features, "wrong number of inputs"
        assert len(X) == len(y), "X and y must have same length"

        y_pred = self.predict(X)
        print(len(y), len(y_pred))
        error = np.sum([((y - y_pred)**2) for y, y_pred in zip(y, y_pred)])
        return error / 2*len(X)


    def _reset_grad(self):
        for w in self.weights:
            w.grad = 0