import numpy as np

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.epochs = epochs
        self.w = None
        self.b = 0

    def fit(self, X, y):
        """
        Args:
            X (python list, (n_samples x n_features)): features matrix
            y (python list, (n_samples)): target vector with labels +1 or -1
        """

        # Parameter check
        if (X is None) or (y is None):
            raise ValueError("X and y must not be None")
        if len(X) == 0 or len(y) == 0:
            raise ValueError("X and y must not be empty")
        if len(X) != len(y):
            raise ValueError("X and y must have same length")


        n_samples = len(X)
        n_features = len(X[0])
        self.w = np.zeros(n_features)  # Initialize weights

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                # Check if the sample is correctly classified
                condition = y[idx] * (np.dot(x_i, self.w) + self.b) >= 1
                if condition:
                    # Correctly classified, only apply regularization
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # Misclassified, update weights and bias
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y[idx]))
                    self.b -= self.learning_rate * y[idx]

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)