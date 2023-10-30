import numpy as np
from tqdm import tqdm
from .utils import *


class NeuralNetwork:
    activation_functions = {
        'relu': relu,
        'sigmoid': sigmoid,
        'tanh': tanh,
        'softmax': softmax,
    }

    loss_functions = {
        'categorical_crossentropy': categorical_crossentropy,
    }

    activation_functions_derivative = {
        'relu': relu_derivative,
        'sigmoid': sigmoid_derivative,
        'tanh': tanh_derivative,
    }

    loss_functions_derivative = {
        'categorical_crossentropy': categorical_crossentropy_derivative,
    }

    def __init__(
        self,
        input_size,
        output_size,
        hidden_layers,
        learning_rate,
        activation_functions,
        loss_function,
        initializer="he",
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate

        for activation_function in activation_functions:
            if activation_function not in self.activation_functions:
                raise ValueError("Invalid activation function")

        if loss_function not in self.loss_functions:
            raise ValueError("Invalid loss function")

        self.activation_functions = [
            self.activation_functions[activation_function]
            for activation_function in activation_functions
        ]
        self.activation_functions_derivative = [
            self.activation_functions_derivative[activation_function]
            for activation_function in activation_functions[:-1]
        ]

        self.loss_function = self.loss_functions[loss_function]
        self.loss_function_derivative = self.loss_functions_derivative[loss_function]

        self.initialize_parameters(initializer)

    def initialize_parameters(self, initializer):
        weights = []
        biases = []

        if initializer == "zeros":
            weights.append(np.zeros((self.input_size, self.hidden_layers[0])))
            biases.append(np.zeros((1, self.hidden_layers[0])))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.zeros((self.hidden_layers[i - 1], self.hidden_layers[i]))
                )
                biases.append(np.zeros((1, self.hidden_layers[i])))

            weights.append(np.zeros((self.hidden_layers[-1], self.output_size)))
            biases.append(np.zeros((1, self.output_size)))

            self.weights = weights
            self.biases = biases

        elif initializer == "ones":
            weights.append(np.ones((self.input_size, self.hidden_layers[0])))
            biases.append(np.ones((1, self.hidden_layers[0])))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.ones((self.hidden_layers[i - 1], self.hidden_layers[i]))
                )
                biases.append(np.ones((1, self.hidden_layers[i])))

            weights.append(np.ones((self.hidden_layers[-1], self.output_size)))
            biases.append(np.ones((1, self.output_size)))

            self.weights = weights
            self.biases = biases

        elif initializer == "random_root":
            factor = 1.0 / np.sqrt(self.input_size)
            weights.append(
                np.random.rand(self.input_size, self.hidden_layers[0]) * factor
            )
            biases.append(np.random.rand(1, self.hidden_layers[0]) * factor)

            for i in range(1, len(self.hidden_layers)):
                factor = 1.0 / np.sqrt(self.hidden_layers[i - 1])
                weights.append(
                    np.random.rand(self.hidden_layers[i - 1], self.hidden_layers[i])
                    * factor
                )
                biases.append(np.random.rand(1, self.hidden_layers[i]) * factor)

            factor = 1.0 / np.sqrt(self.hidden_layers[-1])
            weights.append(
                np.random.rand(self.hidden_layers[-1], self.output_size) * factor
            )
            biases.append(np.random.rand(1, self.output_size) * factor)

            self.weights = weights
            self.biases = biases

        elif initializer == "random_uniform":
            weights.append(
                np.random.uniform(
                    -0.01, 0.01, size=(self.input_size, self.hidden_layers[0])
                )
            )
            biases.append(np.zeros((1, self.hidden_layers[0])))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.random.uniform(
                        -0.01,
                        0.01,
                        size=(self.hidden_layers[i - 1], self.hidden_layers[i]),
                    )
                )
                biases.append(np.zeros((1, self.hidden_layers[i])))

            weights.append(
                np.random.uniform(
                    -0.01, 0.01, size=(self.hidden_layers[-1], self.output_size)
                )
            )
            biases.append(np.zeros((1, self.output_size)))

            self.weights = weights
            self.biases = biases

        elif initializer == "random_normal":
            weights.append(np.random.randn(self.input_size, self.hidden_layers[0]))
            biases.append(np.random.randn(1, self.hidden_layers[0]))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i])
                )
                biases.append(np.random.randn(1, self.hidden_layers[i]))

            weights.append(np.random.randn(self.hidden_layers[-1], self.output_size))
            biases.append(np.random.randn(1, self.output_size))

            self.weights = weights
            self.biases = biases

        elif initializer == "he":
            weights.append(
                np.random.randn(self.input_size, self.hidden_layers[0])
                * np.sqrt(2.0 / self.input_size)
            )
            biases.append(np.zeros((1, self.hidden_layers[0])))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i])
                    * np.sqrt(2.0 / self.hidden_layers[i - 1])
                )
                biases.append(np.zeros((1, self.hidden_layers[i])))

            weights.append(
                np.random.randn(self.hidden_layers[-1], self.output_size)
                * np.sqrt(2.0 / self.hidden_layers[-1])
            )
            biases.append(np.zeros((1, self.output_size)))

            self.weights = weights
            self.biases = biases

        elif initializer == "xavier":
            weights.append(
                np.random.randn(self.input_size, self.hidden_layers[0])
                * np.sqrt(1.0 / self.input_size)
            )
            biases.append(np.zeros((1, self.hidden_layers[0])))

            for i in range(1, len(self.hidden_layers)):
                weights.append(
                    np.random.randn(self.hidden_layers[i - 1], self.hidden_layers[i])
                    * np.sqrt(1.0 / self.hidden_layers[i - 1])
                )
                biases.append(np.zeros((1, self.hidden_layers[i])))

            weights.append(
                np.random.randn(self.hidden_layers[-1], self.output_size)
                * np.sqrt(1.0 / self.hidden_layers[-1])
            )
            biases.append(np.zeros((1, self.output_size)))

            self.weights = weights
            self.biases = biases

        else:
            raise ValueError("Invalid initializer")

    def forward_propagation(self, X):
        layer = X
        cache = {"A0": X}
        for i in range(len(self.weights)):
            activation_function = self.activation_functions[i]
            layer = activation_function(np.dot(layer, self.weights[i]) + self.biases[i])
            cache["A" + str(i + 1)] = layer
        return layer, cache

    def one_hot_encode(self, y):
        one_hot = np.zeros((y.size, self.output_size))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def backward_propagation(self, X, y, y_pred, cache):
        m = X.shape[0]
        gradients = {}

        dZ = self.loss_function_derivative(y_pred, y)

        # Loop through layers in reverse order starting from the output
        for i in reversed(range(len(self.weights))):
            dW = (1 / m) * np.dot(cache["A" + str(i)].T, dZ)
            db = (1 / m) * np.sum(dZ, axis=0, keepdims=True)

            gradients["dW" + str(i + 1)] = dW
            gradients["db" + str(i + 1)] = db

            if i != 0:
                dA_prev = np.dot(dZ, self.weights[i].T)
                dZ = dA_prev * self.activation_functions_derivative[i - 1](
                    cache["A" + str(i)]
                )

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * gradients["dW" + str(i + 1)]
            self.biases[i] -= self.learning_rate * gradients["db" + str(i + 1)]

    def swap_train_data(self, X, y):
        # Swap data
        X, y = X.copy(), y.copy()
        # Shuffle data
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        return X, y

    def train(self, X, y, epochs, batch_size=None):
        y = self.one_hot_encode(y)

        if batch_size is None or batch_size > X.shape[0]:
            batch_size = X.shape[0]

        num_batches = int(X.shape[0] / batch_size)

        self.loss_history = []
        self.accuracy_history = []

        for _ in tqdm(range(epochs)):
            X, y = self.swap_train_data(X, y)
            for j in range(num_batches):
                start = j * batch_size
                end = (j + 1) * batch_size

                X_mini = X[start:end]
                y_mini = y[start:end]

                y_pred, cache = self.forward_propagation(X_mini)
                self.backward_propagation(X_mini, y_mini, y_pred, cache)

                loss = self.loss_function(y_pred, y_mini)
                self.loss_history.append(loss)

                accuracy = self.accuracy(X_mini, np.argmax(y_mini, axis=1))
                self.accuracy_history.append(accuracy)

    def predict(self, X):
        y_pred, _ = self.forward_propagation(X)
        return np.argmax(y_pred, axis=1)

    def accuracy(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y) * 100

    def history(self):
        return self.loss_history, self.accuracy_history
