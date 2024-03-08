#!/usr/bin/env python3
import torch

from lib.value import Value
from lib.nn import MultiLayerPerceptron
from utils import render_expression


def simple_expression():
    a = Value(2.0, _name="a")
    b = Value(-3.0, _name="b")
    c = Value(10.0, _name="c")
    e = a * b
    e._name = "e"
    d = e + c
    d._name = "d"
    f = Value(-2.0, _name="f")
    L = d * f  # L is the loss
    L._name = "L"

    render_expression(L).view()


def weights_expression():
    # Inputs x1, x2
    x1 = Value(2.0, _name="x1")
    x2 = Value(0.0, _name="x2")

    # Weights
    w1 = Value(-3.0, _name="w1")
    w2 = Value(1.0, _name="w2")

    # Value of the neuron is bias + sum[i=1..n] xi*wi
    # Bias of the neuron, this value makes things derivatives nice
    b = Value(6.8813735870195432, _name="b")

    # Construct the neuron's value
    x1w1 = x1 * w1
    x1w1._name = "x1w1"
    x2w2 = x2 * w2
    x2w2._name = "x2w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2._name = "x1w1 + x2w2"
    n = x1w1x2w2 + b
    n.label = "n"

    # Apply an activation energy function to get the final output
    o = n.tanh()
    o.label = "o"

    o.backprop()

    render_expression(o).view()

    o.zero_grad()  # clear the gradients
    render_expression(o).view()


def weights_expression_expanded():
    # Inputs x1, x2
    x1 = Value(2.0, _name="x1")
    x2 = Value(0.0, _name="x2")

    # Weights
    w1 = Value(-3.0, _name="w1")
    w2 = Value(1.0, _name="w2")

    # Value of the neuron is bias + sum[i=1..n] xi*wi
    # Bias of the neuron, this value makes things derivatives nice
    b = Value(6.8813735870195432, _name="b")

    # Construct the neuron's value
    x1w1 = x1 * w1
    x1w1._name = "x1w1"
    x2w2 = x2 * w2
    x2w2._name = "x2w2"
    x1w1x2w2 = x1w1 + x2w2
    x1w1x2w2._name = "x1w1 + x2w2"
    n = x1w1x2w2 + b
    n._name = "n"

    # Apply an activation energy function to get the final output, tanh, but expand it out
    e = (2 * n).exp()
    e._name = "e"
    o = (e - 1) / (e + 1)
    o._name = "o"

    o.backprop()

    render_expression(o).view()


def multivariate_expression():
    a = Value(3.0, _name="a")
    b = a + a

    b.backprop()
    render_expression(b).view()


def pytorch_expression():
    # Leaf nodes usually don't have gradients, so set them explicitly
    x1 = torch.Tensor([2.0]).double()
    x1.requires_grad = True
    x2 = torch.Tensor([0.0]).double()
    x2.requires_grad = True
    w1 = torch.Tensor([-3.0]).double()
    w1.requires_grad = True
    w2 = torch.Tensor([1.0]).double()
    w2.requires_grad = True
    b = torch.Tensor([6.8813735870195432]).double()
    b.requires_grad = True
    n = x1 * w1 + x2 * w2 + b
    o = torch.tanh(n)

    print(o.data.item())
    o.backward()

    print("---")
    print("x2", x2.grad.item())
    print("w2", w2.grad.item())
    print("x1", x1.grad.item())
    print("w1", w1.grad.item())


def simple_mlp_example():
    # This is a 3 layer MLP, with 1 input layer and [2 hidden layers, 1 output layer]
    # This MLP has 3 layers of inputs and outputs
    #   - Layer 1 has 3 inputs and 4 outputs
    #   - Layer 2 has 4 inputs and 4 outputs
    #   - Layer 3 has 4 inputs and 1 output
    x = [2.0, 3.0, -1.0]
    mlp = MultiLayerPerceptron(3, [4, 4, 1])
    print(f"Output of MLP is: {mlp(x)}")
    render_expression(mlp(x)).view()


def mlp_binary_classifier_example():
    # This is a 3 layer MLP, with 1 input layer and [2 hidden layers, 1 output layer]
    # This MLP has 3 layers of inputs and outputs
    #   - Layer 1 has 3 inputs and 4 outputs
    #   - Layer 2 has 4 inputs and 4 outputs
    #   - Layer 3 has 4 inputs and 1 output
    mlp = MultiLayerPerceptron(3, [4, 4, 1])

    xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    ytgts = [1.0, -1.0, -1.0, 1.0]  # desired targets
    # ypreds = [mlp(x) for x in xs]
    # print(f"The predictions are:\n{ypreds}")

    # # Run one iteration of backpropagation
    # loss.backprop()

    learn_rate = 0.01
    learn_steps = 20

    for i in range(learn_steps):
        # Make predictions
        ypreds = [mlp(x) for x in xs]

        # Calculate loss
        loss = sum((ytgt - ypred) ** 2 for ytgt, ypred in zip(ytgts, ypreds))

        # Clear out the gradients
        mlp.zero_grad()

        # Backpropagate
        loss.backprop()

        # Adjust weights
        mlp.update(learn_rate)

        print(f"{i}: Loss is {loss.data}")

    # Print the predictions
    print(f"The predictions are:\n{ypreds}")
    render_expression(loss).view()


mlp_binary_classifier_example()
