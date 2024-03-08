import random

from lib.value import Value


class Module:
    """
    Mixin for neural networks.
    """

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

    def update(self, learn_rate=0):
        for p in self.parameters():
            p.data -= learn_rate * p.grad


class Neuron(Module):
    def __init__(self, num_inputs: int):
        # Initialize weights to random values
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(num_inputs)]
        self.bias = Value(random.uniform(-1, 1))

    def __call__(self, x):
        # This is the forward propagation
        # weights * x + b
        wx = zip(self.weights, x)
        act = sum((wi * xi for wi, xi in wx), self.bias)
        return act.tanh()

    def parameters(self):
        # Get the parameters
        return self.weights + [self.bias]


class Layer(Module):
    def __init__(self, num_inputs, num_outputs):
        # Create neurons equal to the number of outputs
        # Each neuron has a number of inputs equal to the size of the input layer
        self.neurons = [Neuron(num_inputs) for _ in range(num_outputs)]

    def __call__(self, x):
        # This is the forward propagation
        # Calculate the outputs on a per-neuron basis
        outs = [n(x) for n in self.neurons]

        # Just return 1 value if there's only 1 output, instead of a list of length 1
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MultiLayerPerceptron(Module):
    def __init__(self, num_inputs, outputs_per_layer):
        mlp_sizes = [num_inputs] + outputs_per_layer
        self.layers = [
            Layer(mlp_sizes[i], mlp_sizes[i + 1]) for i in range(len(outputs_per_layer))
        ]

    def __call__(self, x):
        # This is the forward propagation
        # Propagate cumulative values between layers
        # The previous layer's out becomes the next layer's in
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
