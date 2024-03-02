import numpy as np
import scipy.io

class Module():
    def __init__(self):
        pass

    def forward(self, input):
        raise NotImplementedError

    def backward(self, grad_output):
        raise NotImplementedError

class Linear(Module):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.01
        self.biases = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        grad_weights = np.dot(self.input.T, grad_output)
        grad_biases = np.sum(grad_output, axis=0, keepdims=True)

        # Update parameters
        Module.learning_rate = 1E-3
        self.weights -= Module.learning_rate * grad_weights
        self.biases -= Module.learning_rate * grad_biases

        return grad_input

class Sigmoid(Module):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, grad_output):
        return grad_output * self.output * (1 - self.output)

class Network(Module):
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

class MeanErrorLoss:
    def forward(self, pred, target):
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2)

    def backward(self):
        return 2 * (self.pred - self.target) / self.target.size

def train(model, data, labels, num_iterations, learning_rate):
    loss_fn = MeanErrorLoss()
    for i in range(num_iterations):
        pred = model.forward(data)
        loss = loss_fn.forward(pred, labels)

        grad_loss = loss_fn.backward()
        model.backward(grad_loss)

        if i % 100 == 0:
            print(f'Iteration {i}, Loss: {loss:.4f}')

def load_and_inspect_mat(filename):
    mat = scipy.io.loadmat(filename)
    variables = {k: v for k, v in mat.items() if not k.startswith('__')}
    sorted_vars = sorted(variables.items(), key=lambda item: item[1].size, reverse=True)

    data, labels = sorted_vars[0][1], sorted_vars[1][1]
    print(f"Data shape: {data.shape}, Labels shape: {labels.shape}")
    return data, labels

if __name__ == "__main__":
    filename = 'nn_data.mat'
    data, labels = load_and_inspect_mat(filename)

    # Assuming the first largest variable is the data and the second is the labels
    # Adjust the network architecture as needed
    network = Network([
        Linear(data.shape[1], 5),  # Adjust the size according to your dataset
        Sigmoid(),
        Linear(5, labels.shape[1])  # Output layer size matches label dimension
    ])

    train(network, data, labels, num_iterations=2000, learning_rate=1E-3)
